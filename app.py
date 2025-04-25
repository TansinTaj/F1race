from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Path to your model and data
DATA_PATH = os.path.join(os.path.dirname(__file__), "final_data_clean.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models.pkl")

# First, check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    # You could exit here or provide fallback logic

# Load models from the pickle file created in Colab
print("Loading models from Colab-generated pickle file...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract all necessary components
    pitstops_model = model_data['pitstops_model']
    pitlap_model = model_data['pitlap_model']
    tire_model = model_data['tire_model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    selected_features = model_data['selected_features']
    
    # Log what's in the pickle file
    print("Model pickle contains:", list(model_data.keys()))
    
    # These might not be in the saved model data, but we'll check
    if 'event_teams' in model_data:
        event_teams = model_data['event_teams']
    else:
        event_teams = None
        
    if 'event_drivers' in model_data:
        event_drivers = model_data['event_drivers']
    else:
        event_drivers = None
        
    if 'event_conditions' in model_data:
        event_conditions = model_data['event_conditions']
    else:
        event_conditions = None
        
    if 'circuit_info' in model_data:
        circuit_info = model_data['circuit_info']
    else:
        circuit_info = None
        
    if 'feature_means' in model_data:
        feature_means = model_data['feature_means']
    else:
        feature_means = None
        
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    # You could exit here or provide fallback logic

# If we need to load the dataset for some metadata (if not included in the pickle)
try:
    if not feature_means or not circuit_info:
        print("Loading dataset to obtain missing metadata...")
        df = pd.read_csv(DATA_PATH)
        df.drop(columns=["Unnamed: 0", "Position"], inplace=True)
        
        # Add dummy calculations for derived features with reasonable defaults if columns don't exist
        if "deg_slope" in df.columns and "deg_bias" in df.columns:
            df["tyreDegradationPerStint"] = (df["deg_slope"] * df["StintLen"] + df["deg_bias"]) / df["StintLen"]
        else:
            # Default value: 0.03 tire degradation per lap
            df["tyreDegradationPerStint"] = 0.03
            
        if "fuel_slope" in df.columns and "fuel_bias" in df.columns:
            df["fuelConsumptionPerStint"] = df["fuel_slope"] * df["StintLen"] + df["fuel_bias"]
        else:
            # Default value: 2.5 kg fuel per lap
            df["fuelConsumptionPerStint"] = 2.5 * df["StintLen"]
            
        if "CircuitLength" in df.columns:
            df["stintPerformance"] = df["StintLen"] / df["CircuitLength"]
        else:
            # Default value assuming average circuit length of 5km
            df["stintPerformance"] = df["StintLen"] / 5.0
            
        if "meanAirTemp" in df.columns and "meanTrackTemp" in df.columns and "meanHumid" in df.columns:
            df["trackConditionIndex"] = (df["meanAirTemp"] + df["meanTrackTemp"]) - df["meanHumid"]
        else:
            # Default value for track condition index
            df["trackConditionIndex"] = 25.0
        
        if not event_teams:
            event_teams = df[['EventName', 'Team']].drop_duplicates()
        if not event_drivers:
            event_drivers = df[['EventName', 'Team', 'Driver']].drop_duplicates()
        if not event_conditions:
            event_conditions = df[['EventName', 'eventYear', 'meanAirTemp', 'meanTrackTemp', 'Rainfall']].drop_duplicates()
        if not circuit_info:
            circuit_info = df[['EventName', 'CircuitLength']].drop_duplicates()
        if not feature_means:
            # Re-create selected features if needed
            if not selected_features:
                selected_features = [
                    "lapNumberAtBeginingOfStint", "eventYear", "meanHumid", "trackConditionIndex", "Rainfall",
                    "designedLaps", "meanTrackTemp", "fuelConsumptionPerStint", "lag_slope_mean", "bestPreRaceTime",
                    "CircuitLength", "StintLen", "RoundNumber", "stintPerformance", "tyreDegradationPerStint", "meanAirTemp"
                ]
            
            # Make sure all features exist in the dataframe
            for feature in selected_features:
                if feature not in df.columns:
                    # Add default values for missing features
                    if feature == "lag_slope_mean":
                        df[feature] = 0.01  # Small default slope
                    elif feature == "bestPreRaceTime":
                        df[feature] = 90.0  # 1:30 lap time in seconds
                    elif feature == "RoundNumber":
                        df[feature] = 10    # Mid-season default
                    else:
                        df[feature] = 0.0   # Generic default
            
            X = df[selected_features]
            feature_means = X.mean().to_dict()
except Exception as e:
    print(f"Error loading dataset for metadata: {str(e)}")
    # Provide default values for missing data
    feature_means = {
        "lapNumberAtBeginingOfStint": 1.0,
        "eventYear": 2024.0,
        "meanHumid": 50.0,
        "trackConditionIndex": 25.0,
        "Rainfall": 0.0,
        "designedLaps": 60.0,
        "meanTrackTemp": 30.0,
        "fuelConsumptionPerStint": 75.0,
        "lag_slope_mean": 0.01,
        "bestPreRaceTime": 90.0,
        "CircuitLength": 5.0,
        "StintLen": 30.0,
        "RoundNumber": 10.0,
        "stintPerformance": 6.0,
        "tyreDegradationPerStint": 0.03,
        "meanAirTemp": 25.0
    }

# Function to get appropriate tire based on rainfall - Keep identical to Colab version
def get_tire_for_conditions(rainfall, predicted_tire, pit_sequence_index=0):
    """
    Determine appropriate tire based on weather conditions
    
    Parameters:
    - rainfall: Rainfall intensity (0-10)
    - predicted_tire: The tire compound predicted by the model
    - pit_sequence_index: The index of this pit stop in the sequence (0 = first pit stop)
    
    Returns:
    - Appropriate tire compound based on weather conditions
    """
    # Define rainfall thresholds
    DRY_THRESHOLD = 0.2
    INTERMEDIATE_THRESHOLD = 3.0
    
    if rainfall is None or rainfall <= DRY_THRESHOLD:
        # Dry conditions - use model prediction
        return predicted_tire
    
    # Only apply special tire rules to the first pit stop in wet conditions
    if pit_sequence_index == 0:
        if rainfall <= INTERMEDIATE_THRESHOLD:
            # Light rain - use intermediates for first pit stop
            return "Intermediate"
        else:
            # Heavy rain - use full wets for first pit stop
            return "Wet"
    
    # For subsequent pit stops in wet conditions, use model prediction
    return predicted_tire

# Function to get average values for a specific event/team/driver - Modified for derived features
def get_event_data(event_name, year=None, team=None, driver=None):
    """
    Get event data based on filters
    """
    # If we didn't get a valid DataFrame earlier, return default values
    if not feature_means:
        return {feature: 0 for feature in selected_features}
    
    # Try to filter based on metadata if available
    filtered_data = {}
    
    if event_name and circuit_info is not None:
        circuit_rows = circuit_info[circuit_info['EventName'] == event_name]
        if not circuit_rows.empty:
            filtered_data["CircuitLength"] = circuit_rows.iloc[0]['CircuitLength']
    
    if event_name and year and event_conditions is not None:
        condition_rows = event_conditions[(event_conditions['EventName'] == event_name) & 
                                        (event_conditions['eventYear'] == year)]
        if not condition_rows.empty:
            filtered_data["meanAirTemp"] = condition_rows.iloc[0]['meanAirTemp']
            filtered_data["meanTrackTemp"] = condition_rows.iloc[0]['meanTrackTemp']
            filtered_data["Rainfall"] = condition_rows.iloc[0]['Rainfall']
            
            # Calculate derived trackConditionIndex here if we have meanHumid
            if "meanHumid" in feature_means:
                filtered_data["trackConditionIndex"] = (filtered_data["meanAirTemp"] + 
                                                        filtered_data["meanTrackTemp"]) - feature_means["meanHumid"]
    
    # Add fallbacks for derived features
    if "StintLen" in feature_means:
        stint_len = feature_means["StintLen"]
        
        # Add derived features with reasonable defaults if not already present
        if "tyreDegradationPerStint" not in filtered_data:
            filtered_data["tyreDegradationPerStint"] = 0.03 * stint_len
        
        if "fuelConsumptionPerStint" not in filtered_data:
            filtered_data["fuelConsumptionPerStint"] = 2.5 * stint_len
        
        if "stintPerformance" not in filtered_data and "CircuitLength" in filtered_data:
            filtered_data["stintPerformance"] = stint_len / filtered_data["CircuitLength"]
        elif "stintPerformance" not in filtered_data:
            filtered_data["stintPerformance"] = stint_len / feature_means.get("CircuitLength", 5.0)
        
        if "trackConditionIndex" not in filtered_data:
            air_temp = filtered_data.get("meanAirTemp", feature_means.get("meanAirTemp", 25.0))
            track_temp = filtered_data.get("meanTrackTemp", feature_means.get("meanTrackTemp", 30.0))
            humid = feature_means.get("meanHumid", 50.0)
            filtered_data["trackConditionIndex"] = (air_temp + track_temp) - humid
    
    # Fill in any missing values with the means
    for feature in selected_features:
        if feature not in filtered_data:
            filtered_data[feature] = feature_means.get(feature, 0)
    
    return filtered_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        print(f"Received prediction request: {json.dumps(data)}")
        
        # Extract values
        track = data['track']
        year = data['year']
        team = data['team']
        driver = data['driver']
        air_temp = data['airTemp']
        track_temp = data['trackTemp']
        rainfall = data['rainfall']
        current_lap = data['currentLap']
        total_laps = data['totalLaps']
        stint_length = data['stintLength']
        
        print(f"DEBUG - Input parameters: track={track}, year={year}, team={team}, driver={driver}")
        print(f"DEBUG - Weather params: air={air_temp}, track={track_temp}, rain={rainfall}")
        print(f"DEBUG - Race params: lap={current_lap}, total={total_laps}, stint={stint_length}")
        
        # Get default values based on track, team, driver
        feature_values = get_event_data(track, year, team, driver)
        print(f"DEBUG - Initial feature values from event data:")
        for k, v in feature_values.items():
            print(f"  {k}: {v}")
        
        # Override with user-provided values
        feature_values["meanAirTemp"] = air_temp
        feature_values["meanTrackTemp"] = track_temp 
        feature_values["Rainfall"] = rainfall
        
        # Recalculate derived features with the new values
        feature_values["trackConditionIndex"] = (air_temp + track_temp) - feature_values["meanHumid"]
        
        # Set race-specific values
        feature_values["lapNumberAtBeginingOfStint"] = current_lap
        feature_values["StintLen"] = stint_length
        feature_values["designedLaps"] = total_laps
        
        # Update derived features after changing StintLen
        if "CircuitLength" in feature_values:
            feature_values["stintPerformance"] = stint_length / feature_values["CircuitLength"]
        feature_values["tyreDegradationPerStint"] = 0.03 * stint_length  # Simple default
        feature_values["fuelConsumptionPerStint"] = 2.5 * stint_length   # Simple default
        
        # Log final feature values after overrides
        print("\nDEBUG - Final feature values after overrides:")
        for k, v in feature_values.items():
            print(f"  {k}: {v}")
        
        # Create feature vector in correct order - EXACTLY LIKE COLAB
        print("\nDEBUG - Creating feature vector in this order:")
        for i, feature in enumerate(selected_features):
            print(f"  {i}: {feature}")
        
        # Match exactly how Colab creates feature vector
        new_input = np.array([feature_values[feature] for feature in selected_features])
        new_df = pd.DataFrame([new_input], columns=selected_features)
        print(f"\nDEBUG - new_df head:")
        print(new_df.head())
        
        # Scale the input
        new_data_scaled = scaler.transform(new_df)
        print(f"\nDEBUG - new_data_scaled shape: {new_data_scaled.shape}")
        print(f"DEBUG - new_data_scaled values: {new_data_scaled}")
        
        # Make predictions
        raw_pitstops_pred = pitstops_model.predict(new_data_scaled)[0]
        pitstops = round(raw_pitstops_pred)
        print(f"\nDEBUG - Pitstops raw prediction: {raw_pitstops_pred}, rounded: {pitstops}")
        
        # Now let's debug each iteration of the loop
        pit_stop_laps = []
        tire_compounds = []
        
        print("\n=== PIT STOP PREDICTION LOOP ===")
        # Use max(1, int(pitstops)) like in Colab to guarantee at least one pit stop
        for i in range(max(1, int(pitstops))):
            print(f"\nDEBUG - Loop iteration {i+1}:")
            print(f"Current DataFrame before prediction:")
            print(new_df.head())
            
            raw_pitlap_pred = pitlap_model.predict(new_data_scaled)[0]
            pitlap = round(raw_pitlap_pred)
            print(f"Raw pitlap prediction: {raw_pitlap_pred}, rounded: {pitlap}")
            
            next_tire_code = tire_model.predict(new_data_scaled)[0]
            print(f"Next tire code prediction: {next_tire_code}")
            
            model_predicted_tire = label_encoders["Compound"].inverse_transform([next_tire_code])[0]
            print(f"Model predicted tire: {model_predicted_tire}")
            
            # Use same weather logic as Colab
            next_tire = get_tire_for_conditions(rainfall, model_predicted_tire, i)
            print(f"Final tire after weather adjustment: {next_tire}")
            
            pit_stop_laps.append(int(pitlap))
            tire_compounds.append(next_tire)
            
            # Update for next iteration - EXACTLY LIKE COLAB
            current_lap = pitlap + 1
            print(f"Updating lap number to: {current_lap}")
            
            # Match Colab's update approach
            new_df["lapNumberAtBeginingOfStint"] = current_lap
            
            # Re-scale
            new_data_scaled = scaler.transform(new_df)
            print("DataFrame after update:")
            print(new_df.head())
        
        # Create result
        result = {
            'pitstops': int(pitstops),
            'pit_lap_sequence': pit_stop_laps,
            'tire_sequence': tire_compounds,
            'weather': {
                'airTemp': air_temp,
                'trackTemp': track_temp,
                'rainfall': rainfall
            }
        }
        
        print(f"\nFinal prediction result: {json.dumps(result)}")
        
        # Only one return statement for successful result
        return jsonify(result)

    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True)