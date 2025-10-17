import sys
import os

print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))
print("Python path:", sys.path)

# Then try the import with explicit path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils_modeling import (
        downcast_numeric_col,
        handle_date_anom_1000,
        handle_date_anom_50,
        convert_days_to_years,
        feature_engineering,
        preprocess_features,
        engineer_features_from_table_enhanced,
        create_installment_custom_features,
        create_bureau_custom_features,
        create_cc_balance_custom_features,
        create_pos_cash_custom_features,
        create_prev_app_custom_features_balanced,
        create_bureau_balance_custom_features,
        engineer_features_from_table,
        ins_num_aggs,
        bureau_num_aggs,
        bb_num_aggs,
        cc_num_aggs,
        prev_num_aggs,
        pos_num_aggs,
    )
    print("Successfully imported utils_modeling")
except ImportError as e:
    print("Import error:", e)
    print("Trying alternative import...")
    # Try absolute import
    try:
        import utils_modeling
        print("Absolute import worked")
    except ImportError as e2:
        print("Absolute import also failed:", e2)
        # Create minimal fallback functions
        def downcast_numeric_col(df):
            return df
        def handle_date_anom_1000(df, cols):
            return df
        # ... create minimal versions of all required functions
        print("Using fallback functions")



import sys
import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), "../Data"))
from utils_modeling import (
    downcast_numeric_col,
    handle_date_anom_1000,
    handle_date_anom_50,
    convert_days_to_years,
    preprocess_features,
    create_installment_custom_features,
    create_bureau_custom_features,
    create_cc_balance_custom_features,
    create_pos_cash_custom_features,
    create_prev_app_custom_features_balanced,
    create_bureau_balance_custom_features,
    engineer_features_from_table,
    ins_num_aggs,
    bureau_num_aggs,
    bb_num_aggs,
    cc_num_aggs,
    prev_num_aggs,
    pos_num_aggs,
)

app = Flask(__name__)

MODEL_PATH = "voting_weighted_ensemble.joblib"
FEATURES_PATH = "lgbm_importances.csv"

model = joblib.load(MODEL_PATH)
importances = pd.read_csv(FEATURES_PATH, index_col=0)["importance"]
top_features = importances.head(170).index.tolist()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON input with keys for the raw tables needed to compute features.
    Example:
    {
        "application": [...],
        "bureau": [...],
        "bureau_balance": [...],
        "prev_app": [...],
        "installments": [...],
        "credit_card_balance": [...],
        "pos_cash": [...]
    }
    """
    try:
        data = request.get_json()

        bureau_features = pd.DataFrame(columns=["SK_ID_CURR"])
        prev_features = pd.DataFrame(columns=["SK_ID_CURR"])
        installments_features = pd.DataFrame(columns=["SK_ID_CURR"])
        credit_card_balance_features = pd.DataFrame(columns=["SK_ID_CURR"])
        pos_features = pd.DataFrame(columns=["SK_ID_CURR"])

        if not data.get("application"):
            return jsonify({"error": "Application data is required"}), 400       

        app_df = pd.DataFrame(data.get("application", []))
        bureau_df = pd.DataFrame(data.get("bureau", []))
        bureau_balance_df = pd.DataFrame(data.get("bureau_balance", []))
        prev_df = pd.DataFrame(data.get("prev_app", []))
        installments_df = pd.DataFrame(data.get("installments", []))
        cc_balance_df = pd.DataFrame(data.get("credit_card_balance", []))
        pos_df = pd.DataFrame(data.get("pos_cash", []))

        if "SK_ID_CURR" not in app_df.columns:
            return jsonify({"error": "SK_ID_CURR missing from application data"}), 400

        app_df = preprocess_features(app_df)

        if not bureau_df.empty:
            bureau_df = downcast_numeric_col(bureau_df)
            date_columns_b = ["DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"]
            bureau_df = handle_date_anom_50(bureau_df, date_columns_b)
            bureau_df = convert_days_to_years(bureau_df)

        if not bureau_balance_df.empty:
            bb_agg_features = create_bureau_balance_custom_features(bureau_balance_df)
            if not bureau_df.empty:
                bureau_enriched_df = bureau_df.merge(bb_agg_features, on="SK_ID_BUREAU", how="left")
            else:
                bureau_enriched_df = bb_agg_features
        else:
            bureau_enriched_df = bureau_df

        bureau_features = engineer_features_from_table(
            bureau_enriched_df,
            "SK_ID_CURR",
            num_aggs={**bureau_num_aggs, **bb_num_aggs},
            cat_aggs={},
            custom_funcs=[create_bureau_custom_features],
            fillna=None
        ) if not bureau_enriched_df.empty else pd.DataFrame(columns=["SK_ID_CURR"])

        if not prev_df.empty:
            prev_df = downcast_numeric_col(prev_df)
            date_columns_prev = [
                "DAYS_FIRST_DRAWING",
                "DAYS_FIRST_DUE",
                "DAYS_LAST_DUE_1ST_VERSION",
                "DAYS_LAST_DUE",
                "DAYS_TERMINATION",
            ]
            prev_df = handle_date_anom_1000(prev_df, date_columns_prev)
            prev_df = convert_days_to_years(prev_df)
            prev_features = engineer_features_from_table(
                prev_df,
                "SK_ID_CURR",
                num_aggs=prev_num_aggs,
                cat_aggs={},
                custom_funcs=[create_prev_app_custom_features_balanced],
                fillna=None
            )

        if not installments_df.empty:
            installments_df = downcast_numeric_col(installments_df)
            installments_features = engineer_features_from_table(
                installments_df,
                "SK_ID_CURR",
                num_aggs=ins_num_aggs,
                cat_aggs={},
                custom_funcs=[create_installment_custom_features],
                fillna=None
            )

        if not cc_balance_df.empty:
            cc_balance_df = downcast_numeric_col(cc_balance_df)
            credit_card_balance_features = engineer_features_from_table(
                cc_balance_df,
                "SK_ID_CURR",
                num_aggs=cc_num_aggs,
                cat_aggs={},
                custom_funcs=[create_cc_balance_custom_features],
                fillna=None
            )

        if not pos_df.empty:
            pos_df = downcast_numeric_col(pos_df)
            pos_features = engineer_features_from_table(
                pos_df,
                "SK_ID_CURR",
                num_aggs=pos_num_aggs,
                cat_aggs={},
                custom_funcs=[create_pos_cash_custom_features],
                fillna=None
            )

        X = app_df.copy()
        for df in [
            bureau_features,
            prev_features,
            installments_features,
            credit_card_balance_features,
            pos_features
        ]:
            if not df.empty and "SK_ID_CURR" in df.columns:
                X = X.merge(df, on="SK_ID_CURR", how="left")

        for f in top_features:
            if f not in X.columns:
                X[f] = np.nan

        X = X[top_features]

        with open("numerical.json") as f:
            numerical = json.load(f)
        with open("categorical.json") as f:
            categorical = json.load(f)

        for c in numerical:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce") 

        for c in categorical:
            if c in X.columns:
                X[c] = X[c].astype(object)

        #if X.isna().any().any():
            #print("Warning: NaN values detected in features")
            #X = X.fillna(0) 

        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        return jsonify({
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "client_id": app_df["SK_ID_CURR"].iloc[0].tolist()
        })

    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        print(error_msg)  # Fallback if logger fails
        if 'logger' in globals():
            logger.error(error_msg)
        return jsonify({"error": error_msg}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

