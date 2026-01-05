import bentoml
import typing as t
import numpy as np
import xgboost as xgb

# Use the name defined in train.py
MODEL_TAG = "airline_satisfaction_model:latest"

# --- Define Bento Service ---
@bentoml.service(name="airline_satisfaction_service", models=[MODEL_TAG])
class AirlineService:
    def __init__(self):
        # We load the model inside __init__ (Best practice for v1.4)
        self.bento_model = bentoml.models.get(MODEL_TAG)
        self.dv = self.bento_model.custom_objects["dv"]
        self.model = bentoml.xgboost.load_model(self.bento_model)
        self.feature_names = self.dv.get_feature_names_out().tolist()
        print(f"Successfully loaded model: {MODEL_TAG}")

    @bentoml.api
    def predict(
        self,
        gender: str,
        customer_type: str,
        age: int,
        type_of_travel: str,
        flight_class: str, # 'class' is reserved, so we use flight_class. 
        flight_distance: int,
        arrival_delay_in_minutes: float,
    ) -> t.Dict[str, t.Any]:
        """
        Predict airline passenger satisfaction.
        """
        # Map arguments to the dictionary keys the model expects
        # Note: We map 'flight_class' back to 'class' because the model was trained on 'class'
        input_data = {
            "gender": gender,
            "customer_type": customer_type,
            "age": age,
            "type_of_travel": type_of_travel,
            "class": flight_class, 
            "flight_distance": flight_distance,
            "arrival_delay_in_minutes": arrival_delay_in_minutes,
        }

        # Transform
        vectorized_data = self.dv.transform([input_data])
        
        # Create DMatrix (Required for XGBoost)
        dinput = xgb.DMatrix(vectorized_data, feature_names=self.feature_names)
        
        # Predict
        prob_satisfied = float(self.model.predict(dinput)[0])

        return {
            "satisfaction_probability": prob_satisfied,
            "is_satisfied": "SATISFIED" if prob_satisfied >= 0.5 else "NEUTRAL/DISSATISFIED",
        }