import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression # <--- Added
from sklearn.metrics import roc_auc_score
import mlflow
import bentoml

# --- Configuration ---
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db" 
BENTO_MODEL_NAME = "airline_satisfaction_model" 
DATA_PATH = "data/train.csv"

def run_training():
    print(f"Starting Training Pipeline...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("airline-satisfaction-bentoml")
    
    with mlflow.start_run():
        
        # --- 1. LOAD & CLEAN DATA ---
        try:
            df = pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            print(f"ERROR: Data file not found at {DATA_PATH}.")
            return

        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df['arrival_delay_in_minutes'] = df['arrival_delay_in_minutes'].fillna(df['arrival_delay_in_minutes'].median())
        
        if df['satisfaction'].dtype == 'object':
            df['satisfaction'] = (df['satisfaction'] == 'satisfied').astype(int)

        df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
        y_full_train = df_full_train.satisfaction.values
        y_test = df_test.satisfaction.values
        
        del df_full_train['satisfaction']
        del df_test['satisfaction']

        # --- 2. PREPROCESSOR ---
        dv = DictVectorizer(sparse=False)
        dict_full_train = df_full_train.to_dict(orient='records')
        X_full_train = dv.fit_transform(dict_full_train)
        
        dict_test = df_test.to_dict(orient='records')
        X_test = dv.transform(dict_test)

        # MODEL A: LOGISTIC REGRESSION (Baseline for Rubric)
        print("Training Model 1: Logistic Regression...")
        lr = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
        lr.fit(X_full_train, y_full_train)
        
        y_pred_lr = lr.predict_proba(X_test)[:, 1]
        auc_lr = roc_auc_score(y_test, y_pred_lr)
        print(f"Logistic Regression AUC: {auc_lr:.4f}")
        mlflow.log_metric("lr_auc", auc_lr)

        # MODEL B: XGBOOST
        print("Training Model 2: XGBoost...")
        dtrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names_out().tolist())
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=dv.get_feature_names_out().tolist())

        xgb_params = {
            'eta': 0.3, 
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': 8,
            'seed': 1,
        }
        
        mlflow.log_params(xgb_params)
        model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=100)
        
        y_pred_xgb = model_xgb.predict(dtest)
        auc_xgb = roc_auc_score(y_test, y_pred_xgb)
        print(f"XGBoost AUC: {auc_xgb:.4f}")
        mlflow.log_metric("xgb_auc", auc_xgb)
        

        # SAVE THE BEST MODEL TO BENTOML
        # XGBoost is better, so we save that one for the app.
        
        bento_model = bentoml.xgboost.save_model(
            BENTO_MODEL_NAME, 
            model_xgb,
            custom_objects={"dv": dv}
        )
        print(f"\nBentoML Model Saved: {bento_model.tag}")
        mlflow.set_tag("bentoml_model_tag", bento_model.tag)

if __name__ == '__main__':
    run_training()