import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Callable
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def handle_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles anomalies in specific numerical features by capping extreme values
    and creating new binary anomaly flags.

    This function performs targeted anomaly handling for several columns:
    
    - **`CNT_CHILDREN` & `CNT_FAM_MEMBERS`**: Caps the number of children and
      family members at realistic maximums (7 and 9, respectively) and creates
      binary flags to identify these unusual cases.
    
    - **`AMT_INCOME_TOTAL`**: Replaces extremely high income values (>= 1e8)
      with the median of the non-extreme values to mitigate their skewing effect.
    
    - **Social Circle Features**: For features related to social circle
      observations and defaults, values exceeding the 99.9th percentile are
      capped at that percentile, and corresponding anomaly flags are created.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with handled anomalies and new anomaly flags.
    """
       
    if "CNT_CHILDREN" in df.columns:
        df["CHILDREN_ANOMALY"] = (df["CNT_CHILDREN"] > 6).astype(int)
        df.loc[df["CNT_CHILDREN"] > 7, "CNT_CHILDREN"] = 7

    if "CNT_FAM_MEMBERS" in df.columns:
        df["FAM_MEMBERS_ANOMALY"] = (df["CNT_FAM_MEMBERS"] > 8).astype(int)
        df.loc[df["CNT_FAM_MEMBERS"] > 9, "CNT_FAM_MEMBERS"] = 9

    
    if "AMT_INCOME_TOTAL" in df.columns:
        extreme_threshold = 1e8  
        median_val = df.loc[
            df["AMT_INCOME_TOTAL"] < extreme_threshold, "AMT_INCOME_TOTAL"
        ].median()
        df.loc[df["AMT_INCOME_TOTAL"] >= extreme_threshold, "AMT_INCOME_TOTAL"] = (
            median_val
        )

    
    for col in [
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
    ]:
        if col in df.columns:
            cap_value = df[col].quantile(0.999)
            df[f"{col}_ANOMALY"] = (df[col] > cap_value).astype(int)
            df.loc[df[col] > cap_value, col] = cap_value

    return df


def collapse_building_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidates individual building-related missing flags into two new features
    and drops the original columns.
    
    This function identifies all columns that act as missing value flags for
    building-related features (those ending in '_MISSING' and containing 'AVG').
    It then creates two new, more informative features:
        
    - 'BUILDING_INFO_ANY_MISSING': A binary flag (1 or 0) indicating whether
        any building-related information is missing for a client.
    - 'BUILDING_INFO_MISSING_COUNT': A count of how many building-related
        features are missing for each client.
          
    This approach simplifies the feature set by replacing multiple highly
    correlated binary flags with two consolidated features. The original
    individual flags are then dropped.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with the new consolidated features and
                          the original flags removed.
        
    """
    building_flags = [
        col for col in df.columns if col.endswith("_MISSING") and "AVG" in col
    ]
    if building_flags:
        df["BUILDING_INFO_ANY_MISSING"] = df[building_flags].max(axis=1)
        df["BUILDING_INFO_MISSING_COUNT"] = df[building_flags].sum(axis=1)
       
        df = df.drop(columns=building_flags)

    return df


 
    

def create_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary missing value flags for specified columns and consolidates
    building-related features by dropping redundant columns and creating
    missing flags for the remaining ones.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new missing flag columns and
                      redundant building-related columns dropped.
    """
    for col in ["EXT_SOURCE_1", "OCCUPATION_TYPE"]:
        if col in df.columns:
            df[col + "_MISSING"] = df[col].isna().astype(int)

        # Collapse building flags into _AVG only (drop MEDI, MODE)
    redundant_groups = [
        ["COMMONAREA_AVG", "COMMONAREA_MEDI", "COMMONAREA_MODE"],
        [
            "NONLIVINGAPARTMENTS_AVG",
            "NONLIVINGAPARTMENTS_MEDI",
            "NONLIVINGAPARTMENTS_MODE",
        ],
        ["LIVINGAPARTMENTS_AVG", "LIVINGAPARTMENTS_MEDI", "LIVINGAPARTMENTS_MODE"],
        ["YEARS_BUILD_AVG", "YEARS_BUILD_MEDI", "YEARS_BUILD_MODE"],
        ["LANDAREA_AVG", "LANDAREA_MEDI", "LANDAREA_MODE"],
        ["BASEMENTAREA_AVG", "BASEMENTAREA_MEDI", "BASEMENTAREA_MODE"],
        ["NONLIVINGAREA_AVG", "NONLIVINGAREA_MEDI", "NONLIVINGAREA_MODE"],
        ["ELEVATORS_AVG", "ELEVATORS_MEDI", "ELEVATORS_MODE"],
        ["APARTMENTS_AVG", "APARTMENTS_MEDI", "APARTMENTS_MODE"],
        ["ENTRANCES_AVG", "ENTRANCES_MEDI", "ENTRANCES_MODE"],
        ["LIVINGAREA_AVG", "LIVINGAREA_MEDI", "LIVINGAREA_MODE"],
        ["FLOORSMAX_AVG", "FLOORSMAX_MEDI", "FLOORSMAX_MODE"],
        ["FLOORSMIN_AVG", "FLOORSMIN_MEDI", "FLOORSMIN_MODE"],
        [
            "YEARS_BEGINEXPLUATATION_AVG",
            "YEARS_BEGINEXPLUATATION_MEDI",
            "YEARS_BEGINEXPLUATATION_MODE",
        ],
    ]
    drop_triplets = [col for group in redundant_groups for col in group[1:]]
    df = df.drop(
        columns=[c for c in drop_triplets if c in df.columns], errors="ignore"
    )

    avg_columns = [col for group in redundant_groups for col in group[:1]]
    for col in avg_columns:
        if col in df.columns:
            df[col + "_MISSING"] = df[col].isna().astype(int)

    return df


def handle_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces rare or undefined categories in specific columns with a common 'Other' value.

    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with specified rare categories handled.
    """
    rare_replace = {
        "CODE_GENDER": {"XNA": "Other"},
        "NAME_FAMILY_STATUS": {"Unknown": "Other"},
        "NAME_INCOME_TYPE": {"Maternity leave": "Other"},
    }
    for col, mapping in rare_replace.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df






def downcast_numeric_col(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function for downcasting numeric columns in data frame.
    data: array-like dataset,
    Returns: downcasted array-like dataset.
    """
    for col in data.columns:
        if pd.api.types.is_integer_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="integer")
        elif pd.api.types.is_float_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="float")
    return data


def handle_date_anom_1000(df, date_columns)-> pd.DataFrame:
    """
    Handles the 1000-year date anomalies across datasets
    """
    df = df.copy()

    for col in date_columns:
        if col in df.columns:
            df[f"ANOM_{col}"] = (df[col] == 365243).astype(int)
            df[col] = df[col].replace({365243: np.nan})


    return df


def handle_date_anom_50(df, date_columns)-> pd.DataFrame:
    """
    Handles the more 50-year date anomalies across all datasets
    """
    df = df.copy()

    for col in date_columns:
        if col in df.columns:
            df.loc[(df[col] < -18250) | (df[col] > 18250), col] = np.nan

    return df


def convert_days_to_years(df: pd.DataFrame, negative=True) -> pd.DataFrame:
    """
    Converts columns prefixed with "DAYS_" from days to years.

    Args:
        df (pd.DataFrame): The input DataFrame.
        negative (bool): If True, converts negative days to positive years.
                         If False, converts positive days to positive years.

    Returns:
        pd.DataFrame: A new DataFrame with the converted columns.
    """

    df = df.copy()

    days_cols = df.columns[df.columns.str.startswith("DAYS_")]

    sign = -1 if negative else 1

    df[days_cols] = df[days_cols] * sign / 365.25

    new_col_names = [col.replace("DAYS_", "YEARS_") for col in days_cols]
    df.rename(columns=dict(zip(days_cols, new_col_names)), inplace=True)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
        Creates new features (full description in EDA Feature Engineering part):
        1. Financial Ratios
        2. Asset Ownership
        3. Contactability / Phone change feature
        4. External Scores (EXT_SOURCE)
        5. Regional and City Stability
        6. Application Timing
        7. Age features
        8. Document Provision Score / Any Document Flag
        9. Social Circle Default Rates
        10. Registration vs. ID Publish Difference (REGISTRATION_ID_DIFF)
        11. Housing Quality Score (HOUSING_QUALITY_SCORE)
        12. Single With Children (SINGLE_WITH_CHILDREN)
        
        Args:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with engineered new features.
    """

    new_features = {}
        
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        ratio1 = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        new_features["DTI_RATIO"] = ratio1.replace([np.inf, -np.inf], np.nan)

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        ratio2 = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        new_features["CREDIT_TO_INCOME"] = ratio2.replace([np.inf, -np.inf], np.nan)

    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        ratio3 = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
        new_features["ANNUITY_TO_CREDIT"] = ratio3.replace(
            [np.inf, -np.inf], np.nan
        )

    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(df.columns):
        ratio4 = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
        ratio4 = ratio4.replace([np.inf, -np.inf], np.nan)
        new_features["LTV_RATIO"] = ratio4.clip(0, 2)

    if {"AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        ratio5 = df["AMT_GOODS_PRICE"] / df["AMT_INCOME_TOTAL"]
        new_features["GOODS_TO_INCOME"] = ratio5.replace([np.inf, -np.inf], np.nan)

        
    if {"FLAG_OWN_CAR", "FLAG_OWN_REALTY"}.issubset(df.columns):
        new_features["ASSET_SCORE"] = df["FLAG_OWN_CAR"] + df["FLAG_OWN_REALTY"]

    
    phone_flags = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_PHONE"]
    available = [c for c in phone_flags if c in df.columns]
    if available:
        new_features["PHONE_PROVIDED"] = df[available].sum(axis=1).clip(0, 1)

    if "YEARS_LAST_PHONE_CHANGE" in df.columns:
        new_features["PHONE_CHANGED_RECENTLY"] = (
            df["YEARS_LAST_PHONE_CHANGE"] < 1
        ).astype(int)

       
    ext_cols = [
        c
        for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        if c in df.columns
    ]
    for col in ext_cols:
        df[f"{col}_SQ"] = df[col] ** 2

    if len(ext_cols) == 3:
        new_features["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        new_features["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
        new_features["EXT_SOURCE_1_2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
        new_features["EXT_SOURCE_1_3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
        new_features["EXT_SOURCE_2_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    
    region_cols = [
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
    ]
    if all(col in df.columns for col in region_cols):
        new_features["REGION_MISMATCH_SUM"] = df[region_cols].sum(axis=1)
        new_features["REGION_MISMATCH_FLAG"] = (
            df[region_cols].sum(axis=1) > 0
        ).astype(int)

    city_cols = [
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ]
    if all(col in df.columns for col in city_cols):
        new_features["CITY_MISMATCH_SUM"] = df[city_cols].sum(axis=1)

        
    if "WEEKDAY_APPR_PROCESS_START" in df.columns:
        df = pd.get_dummies(
            df, columns=["WEEKDAY_APPR_PROCESS_START"], prefix="APP_WEEKDAY"
        )

    if "HOUR_APPR_PROCESS_START" in df.columns:
        conditions = [
            (df["HOUR_APPR_PROCESS_START"] >= 8)
            & (df["HOUR_APPR_PROCESS_START"] < 12),
            (df["HOUR_APPR_PROCESS_START"] >= 12)
            & (df["HOUR_APPR_PROCESS_START"] < 16),
            (df["HOUR_APPR_PROCESS_START"] >= 16)
            & (df["HOUR_APPR_PROCESS_START"] < 20),
        ]
        choices = ["Morning", "Afternoon", "Evening"]
        hour_bucket = np.select(conditions, choices, default="Other")

        hour_dummies = pd.get_dummies(hour_bucket, prefix="APP_HOUR")
        for col in hour_dummies.columns:
            new_features[col] = hour_dummies[col]

        
    if "YEARS_BIRTH" in df.columns:
        bins = [-np.inf, 25, 35, 50, np.inf]
        labels = ["18-25", "26-35", "36-50", "50+"]
        new_features["AGE_GROUP"] = pd.cut(
            df["YEARS_BIRTH"], bins=bins, labels=labels
        )

    if {"YEARS_EMPLOYED", "YEARS_BIRTH"}.issubset(df.columns):
        ratio6 = df["YEARS_EMPLOYED"] / df["YEARS_BIRTH"]
        new_features["EMPLOY_AGE_RATIO"] = ratio6.replace([np.inf, -np.inf], np.nan)

    if {"YEARS_REGISTRATION", "YEARS_BIRTH"}.issubset(df.columns):
        ratio7 = df["YEARS_REGISTRATION"] / df["YEARS_BIRTH"]
        new_features["REGISTRATION_AGE_RATIO"] = ratio7.replace(
            [np.inf, -np.inf], np.nan
        )

    
    document_flags = [col for col in df.columns if "FLAG_DOCUMENT_" in col]
    if document_flags:
        new_features["DOCUMENT_SCORE"] = df[document_flags].sum(axis=1)
        new_features["ANY_DOCUMENT_PROVIDED"] = (
            df[document_flags].sum(axis=1) > 0
        ).astype(int)

    
    if {"DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"}.issubset(
        df.columns
    ):
        ratio8 = df["DEF_30_CNT_SOCIAL_CIRCLE"] / df["OBS_30_CNT_SOCIAL_CIRCLE"]
        new_features["SOCIAL_30_DPD_RATE"] = ratio8.replace(
            [np.inf, -np.inf], np.nan
        )

    if {"DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"}.issubset(
        df.columns
    ):
        ratio9 = df["DEF_60_CNT_SOCIAL_CIRCLE"] / df["OBS_60_CNT_SOCIAL_CIRCLE"]
        new_features["SOCIAL_60_DPD_RATE"] = ratio9.replace(
            [np.inf, -np.inf], np.nan
        )

    
    if {"YEARS_REGISTRATION", "YEARS_ID_PUBLISH"}.issubset(df.columns):
        new_features["REGISTRATION_ID_DIFF"] = (
            df["YEARS_REGISTRATION"] - df["YEARS_ID_PUBLISH"]
        )

    
    housing_cols = [
        c for c in ["NAME_HOUSING_TYPE", "WALLSMATERIAL_MODE"] if c in df.columns
    ]
    if housing_cols:
        new_features["HOUSING_QUALITY_SCORE"] = (
            df[housing_cols]
            .astype("category")
            .apply(lambda x: hash(tuple(x)), axis=1)
        )

    
    if {"NAME_FAMILY_STATUS", "CNT_CHILDREN"}.issubset(df.columns):
        new_features["SINGLE_WITH_CHILDREN"] = (
            (df["NAME_FAMILY_STATUS"] == "Single") & (df["CNT_CHILDREN"] > 0)
        ).astype(int)

    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive preprocessing and feature engineering for Home Credit application data.

    This function prepares the raw application dataset for modeling by performing 
    a sequence of cleaning, anomaly handling, and feature creation steps. 
    The goal is to maximize predictive information while keeping the data consistent 
    and numeric-friendly for machine learning algorithms.

    Steps included:

    1. **Numeric downcasting**
       - Converts numeric columns to the smallest appropriate dtype (int8, float32, etc.) 
         to save memory without losing precision.
       - Speeds up computations for large datasets.

    2. **Date anomalies handling**
       - Detects and fixes unrealistic or placeholder values in DAYS_* columns.
       - Examples:
         - `DAYS_EMPLOYED` with value 365243 (a known placeholder for missing) 
           is replaced with NaN or a corrected value.
         - `DAYS_REGISTRATION` anomalies (>50 years) are capped or flagged.
       
    3. **Convert DAYS to YEARS**
       - Converts relevant features from days to years (e.g., age, employment duration) 
         for more interpretable scales and stability in downstream calculations.

    4. **Rare category handling**
       - Collapses infrequent categories in categorical columns into a single 'RARE' category.
       - Reduces noise and sparsity, improving model stability.

    5. **Missing value flags**
       - Creates binary indicators for missing values in key features.
       - Helps the model detect patterns associated with missing data.

    6. **Building/collapsing flags**
       - Merges multiple related binary or categorical flags into consolidated indicators.
       - Examples: building type flags, property ownership flags.

    7. **Anomaly corrections**
       - Corrects extreme outliers or implausible values in financial, demographic, 
         and derived features.
       - Ensures stability in ratios, differences, and other engineered metrics.

    8. **Predictive feature engineering**
       - Generates new features expected to improve model predictive performance:
         - Ratios, differences, interactions (e.g., credit utilization, income-to-loan ratios)
         - Flag variables for risk, anomalies, or historical behavior
         - Temporal features derived from DAYS/YEARS columns
       - May also include aggregation or transformation of multiple columns into 
         composite indicators.

    **Returns:**
        - `pd.DataFrame`: a cleaned and fully preprocessed DataFrame ready for 
          machine learning, with numeric-friendly types, missing flags, and 
          engineered features included.
    """

    df = downcast_numeric_col(df)
    df = handle_date_anom_1000(df, ["DAYS_EMPLOYED"])
    df = handle_date_anom_50(df, ["DAYS_REGISTRATION"])
    df = convert_days_to_years(df)
    df = handle_rare_categories(df)
    df = create_missing_flags(df)
    df = collapse_building_flags(df)
    df = handle_anomalies(df)
    df = feature_engineering(df)

    return df


def engineer_features_from_table_enhanced(
    main_df: pd.DataFrame,
    auxiliary_df: pd.DataFrame,
    group_id: str,
    num_aggs: Dict[str, List[str]] = {},
    cat_aggs: Dict[str, List[str]] = {},
    custom_funcs: List[Callable[[pd.DataFrame], pd.DataFrame]] = [],
    must_keep_list: List[str] = [],
    selection_method: str = "adaptive",  # "fixed", "percentage", "adaptive"
    n_fixed: int = 15,
    keep_pct: float = 0.3,
    importance_threshold: float = 0.7,
    min_features: int = 5,
    max_features: int = 35,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates aggregated and engineered features from an auxiliary table,
    optionally merges them with a main table, and selects top features based
    on a combination of LightGBM feature importance and ROC-AUC score.

    Parameters
    ----------
    main_df : pd.DataFrame
        The primary dataframe containing the target variable ('TARGET') and group identifiers.
    auxiliary_df : pd.DataFrame
        Auxiliary dataframe used for aggregations and custom feature engineering.
    group_id : str
        Column name used to group `auxiliary_df` for aggregation.
    num_aggs : dict, optional
        Dictionary specifying numeric aggregations: {column_name: [agg_functions]}.
    cat_aggs : dict, optional
        Dictionary specifying categorical aggregations: {column_name: [agg_functions]}.
    custom_funcs : list of callables, optional
        Functions that take `auxiliary_df` and return additional features as a DataFrame
        to merge on `group_id`.
    must_keep_list : list of str, optional
        List of feature names that must be included in the final selection regardless of ranking.
    selection_method : str, default "adaptive"
        Method for selecting top features:
        - "fixed": select a fixed number (`n_fixed`) of top features.
        - "percentage": select a percentage (`keep_pct`) of features within min/max bounds.
        - "adaptive": dynamically adjust selection based on total number of features.
    n_fixed : int, default 15
        Number of features to select if `selection_method="fixed"`.
    keep_pct : float, default 0.3
        Percentage of features to keep if `selection_method="percentage"`.
    importance_threshold : float, default 0.7
        Minimum combined rank threshold to consider (not directly used in current logic, reserved for future).
    min_features : int, default 5
        Minimum number of features to keep.
    max_features : int, default 35
        Maximum number of features to keep.

    Returns
    -------
    selected_features_df : pd.DataFrame
        DataFrame containing `group_id` and selected aggregated/engineered features.
    analysis_df : pd.DataFrame
        DataFrame with detailed feature analysis including LightGBM importance, ROC-AUC, and combined rank.
        If `main_df` does not contain 'TARGET', returns empty DataFrame for analysis_df.
    """
    grp = auxiliary_df.groupby(group_id)

    agg_df = grp.agg({**num_aggs, **cat_aggs})

    agg_df.columns = [f"{col[0]}_{col[1]}" for col in agg_df.columns]
    agg_df = agg_df.reset_index()

    for func in custom_funcs:
        custom_feats = func(auxiliary_df)
        agg_df = agg_df.merge(custom_feats, on=group_id, how="left")

    if "TARGET" in main_df.columns:
        df_for_analysis = main_df[[group_id, "TARGET"]].merge(
            agg_df, on=group_id, how="left"
        )
        features = [col for col in agg_df.columns if col != group_id]

        X = df_for_analysis[features].fillna(0)
        y = df_for_analysis["TARGET"]

        if len(features) > 0 and y.nunique() > 1:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100, random_state=42, verbosity=-1
            )
            lgb_model.fit(X, y)
            lgb_importance = lgb_model.feature_importances_

            roc_auc_scores = []
            for feat in features:
                if len(np.unique(X[feat])) > 1:
                    try:
                        roc_auc = roc_auc_score(y, X[feat])
                    except:
                        roc_auc = 0.5
                else:
                    roc_auc = 0.5
                roc_auc_scores.append(roc_auc)

            analysis_df = pd.DataFrame(
                {
                    "feature": features,
                    "lgb_importance": lgb_importance,
                    "roc_auc": roc_auc_scores,
                }
            )

            analysis_df["rank_importance"] = analysis_df["lgb_importance"].rank(
                ascending=False
            )
            analysis_df["rank_roc_auc"] = analysis_df["roc_auc"].rank(ascending=False)

            analysis_df["combined_rank"] = (
                analysis_df["rank_importance"] * 0.6
                + analysis_df["rank_roc_auc"] * 0.4
            )

            if selection_method == "fixed":
                top_features = analysis_df.nsmallest(n_fixed, "combined_rank")[
                    "feature"
                ].tolist()

            elif selection_method == "percentage":
                n_to_keep = int(len(analysis_df) * keep_pct)
                n_to_keep = max(min_features, min(n_to_keep, max_features))
                top_features = analysis_df.nsmallest(n_to_keep, "combined_rank")[
                    "feature"
                ].tolist()

            elif (
                selection_method == "adaptive"
            ): 
                n_total = len(analysis_df)
                if n_total <= 15:
                    keep_pct_adaptive = 0.9
                elif n_total <= 30:
                    keep_pct_adaptive = 0.7
                else:
                    keep_pct_adaptive = 0.5
                n_to_keep = int(n_total * keep_pct_adaptive)
                n_to_keep = max(min_features, min(n_to_keep, max_features))
                top_features = analysis_df.nsmallest(n_to_keep, "combined_rank")[
                    "feature"
                ].tolist()

            for feat in must_keep_list:
                if feat in analysis_df["feature"].values and feat not in top_features:
                    print(f"Force-adding must-keep feature: {feat}")
                    top_features.append(feat)

            selected_cols = [group_id] + top_features
            selected_features_df = agg_df[selected_cols]

            return selected_features_df, analysis_df.sort_values("combined_rank")

    return agg_df, pd.DataFrame()


def create_installment_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates human-engineered features from an installment payments DataFrame.
    Features capture late payments, underpayment, full repayment, and differences
    between scheduled and actual payments, aggregated by customer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing installment payment records with at least the following columns:
        - 'SK_ID_CURR': customer ID
        - 'DAYS_ENTRY_PAYMENT': days when payment was made
        - 'DAYS_INSTALMENT': days when installment was due
        - 'AMT_INSTALMENT': scheduled installment amount
        - 'AMT_PAYMENT': actual payment amount

    Returns
    -------
    pd.DataFrame
        Aggregated customer-level features with columns like:
        - INSTAL_LATE_DAYS_max, INSTAL_LATE_DAYS_mean
        - INSTAL_LATE_FLAG_mean, INSTAL_LATE_FLAG_sum
        - INSTAL_UNDERPAYMENT_RATIO_min, INSTAL_UNDERPAYMENT_RATIO_mean, INSTAL_UNDERPAYMENT_RATIO_std
        - INSTAL_UNDERPAYMENT_FLAG_sum
        - INSTAL_FULL_REPAYMENT_FLAG_sum
        - INSTAL_PAYMENT_DIFF_mean, INSTAL_PAYMENT_DIFF_sum
        - INSTAL_ABS_PAYMENT_DIFF_mean, INSTAL_ABS_PAYMENT_DIFF_sum

    Notes
    -----
    - 'LATE_DAYS' = abs(DAYS_ENTRY_PAYMENT) - abs(DAYS_INSTALMENT)
    - 'LATE_FLAG' = 1 if payment is late, else 0
    - 'UNDERPAYMENT_RATIO' = AMT_PAYMENT / AMT_INSTALMENT (1 if AMT_INSTALMENT is 0)
    - 'UNDERPAYMENT_FLAG' = 1 if UNDERPAYMENT_RATIO < 0.95
    - 'FULL_REPAYMENT_FLAG' = 1 if UNDERPAYMENT_RATIO >= 1.05
    - 'PAYMENT_DIFF' = AMT_PAYMENT - AMT_INSTALMENT
    - 'ABS_PAYMENT_DIFF' = absolute value of PAYMENT_DIFF
    """
    
    df[["DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"]] = df[
        ["DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"]
    ].abs()

    df["LATE_DAYS"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["LATE_FLAG"] = (df["LATE_DAYS"] > 0).astype(int)
    df["UNDERPAYMENT_RATIO"] = np.where(
        df["AMT_INSTALMENT"] != 0, 
        df["AMT_PAYMENT"] / df["AMT_INSTALMENT"],
        1.0,
    )

    df["UNDERPAYMENT_FLAG"] = (df["UNDERPAYMENT_RATIO"] < 0.95).astype(
        int
    ) 
    df["FULL_REPAYMENT_FLAG"] = (df["UNDERPAYMENT_RATIO"] >= 1.05).astype(int)

    df["PAYMENT_DIFF"] = df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]
    df["ABS_PAYMENT_DIFF"] = abs(df["AMT_PAYMENT"] - df["AMT_INSTALMENT"])

    grp = df.groupby("SK_ID_CURR")

    custom_aggs = {
        "LATE_DAYS": ["max", "mean"],
        "LATE_FLAG": ["mean", "sum"],
        "UNDERPAYMENT_RATIO": ["min", "mean", "std"],
        "UNDERPAYMENT_FLAG": "sum",
        "FULL_REPAYMENT_FLAG": "sum",
        "PAYMENT_DIFF": ["mean", "sum"],
        "ABS_PAYMENT_DIFF": ["mean", "sum"],
    }

    custom_agg_df = grp.agg(custom_aggs)
    custom_agg_df.columns = [
        f"INSTAL_{col[0]}_{col[1]}" for col in custom_agg_df.columns
    ]
    return custom_agg_df.reset_index()


def create_bureau_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates human-engineered features from bureau credit data.
    These features capture the client's overall credit health, credit utilization,
    overdue payments, and major credit risks, aggregated by customer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing bureau credit records with at least the following columns:
        - 'SK_ID_CURR': customer ID
        - 'CREDIT_ACTIVE': status of the credit (e.g., 'Active', 'Closed', 'Sold', 'Bad debt')
        - 'AMT_CREDIT_SUM_LIMIT': credit limit amount
        - 'AMT_CREDIT_SUM_DEBT': current debt amount
        - 'AMT_CREDIT_SUM': total credit sum
        - 'CREDIT_DAY_OVERDUE': number of days credit is overdue
        - 'YEARS_CREDIT_ENDDATE': credit end date in years
        - 'YEARS_CREDIT': credit start date in years
        - 'AMT_CREDIT_SUM_OVERDUE': overdue credit amount
        - 'YEARS_CREDIT_UPDATE': last update of credit in years

    Returns
    -------
    pd.DataFrame
        Aggregated customer-level features with columns like:
        - BUREAU_HAS_ACTIVE_CREDIT_sum
        - BUREAU_HAS_CLOSED_CREDIT_sum
        - BUREAU_HAS_BAD_CREDIT_sum
        - BUREAU_HAS_ANY_OVERDUE_DEBT_max
        - BUREAU_HAS_SIGNIFICANT_OVERDUE_DEBT_max
        - BUREAU_HAS_ANY_MAJOR_BUREAU_RISK_max
        - BUREAU_CREDIT_UTILIZATION_mean, BUREAU_CREDIT_UTILIZATION_max
        - BUREAU_YEARS_CREDIT_UPDATE_min, BUREAU_YEARS_CREDIT_UPDATE_max
        - BUREAU_CREDIT_ENDDATE_PROXIMITY_mean
        - BUREAU_CREDIT_COUNT (total bureau records per customer)
        - BUREAU_ACTIVE_CREDIT_COUNT (total active credits per customer)

    Notes
    -----
    - 'HAS_ACTIVE_CREDIT' = 1 if CREDIT_ACTIVE == "Active", else 0
    - 'HAS_CLOSED_CREDIT' = 1 if CREDIT_ACTIVE == "Closed", else 0
    - 'HAS_BAD_CREDIT' = 1 if CREDIT_ACTIVE in ["Sold", "Bad debt"], else 0
    - Negative values in 'AMT_CREDIT_SUM_LIMIT' and 'AMT_CREDIT_SUM_DEBT' are flagged
      and replaced with 0
    - 'CREDIT_UTILIZATION' = AMT_CREDIT_SUM_DEBT / CURRENT_CREDIT_LIMIT (clipped 0-1)
    - 'DAYS_CREDIT_OVERDUE' = CREDIT_DAY_OVERDUE
    - 'CREDIT_ENDDATE_PROXIMITY' = YEARS_CREDIT_ENDDATE - YEARS_CREDIT
    - 'HAS_ANY_OVERDUE_DEBT' = 1 if AMT_CREDIT_SUM_OVERDUE > 0
    - 'HAS_SIGNIFICANT_OVERDUE_DEBT' = 1 if AMT_CREDIT_SUM_OVERDUE > 1000
    - 'HAS_ANY_MAJOR_BUREAU_RISK' = 1 if HAS_ANY_OVERDUE_DEBT or HAS_BAD_CREDIT
    """
    temp_df = df.copy()

    temp_df["HAS_ACTIVE_CREDIT"] = (temp_df["CREDIT_ACTIVE"] == "Active").astype(int)
    temp_df["HAS_CLOSED_CREDIT"] = (temp_df["CREDIT_ACTIVE"] == "Closed").astype(int)
    temp_df["HAS_BAD_CREDIT"] = (
        temp_df["CREDIT_ACTIVE"].isin(["Sold", "Bad debt"])
    ).astype(int)

    for col in ["AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_DEBT"]:
        temp_df[f"FLAG_NEG_{col}"] = (temp_df[col] < 0).astype(int)
        temp_df[col] = temp_df[col].clip(lower=0)

    temp_df["CURRENT_CREDIT_LIMIT"] = temp_df["AMT_CREDIT_SUM"].fillna(0)
    temp_df["CREDIT_UTILIZATION"] = np.where(
        temp_df["CURRENT_CREDIT_LIMIT"] > 0,
        temp_df["AMT_CREDIT_SUM_DEBT"] / temp_df["CURRENT_CREDIT_LIMIT"],
        0,
    )

    temp_df["CREDIT_UTILIZATION"] = temp_df["CREDIT_UTILIZATION"].clip(0, 1)

    temp_df["DAYS_CREDIT_OVERDUE"] = temp_df["CREDIT_DAY_OVERDUE"]
    temp_df.drop(columns="CREDIT_DAY_OVERDUE", inplace=True)

    temp_df["CREDIT_ENDDATE_PROXIMITY"] = (
        temp_df["YEARS_CREDIT_ENDDATE"] - temp_df["YEARS_CREDIT"]
    )

    temp_df["HAS_ANY_OVERDUE_DEBT"] = (temp_df["AMT_CREDIT_SUM_OVERDUE"] > 0).astype(
        int
    )
    temp_df["HAS_SIGNIFICANT_OVERDUE_DEBT"] = (
        temp_df["AMT_CREDIT_SUM_OVERDUE"] > 1000
    ).astype(int)

    temp_df["HAS_ANY_MAJOR_BUREAU_RISK"] = (
        (temp_df["HAS_ANY_OVERDUE_DEBT"] > 0) | (temp_df["HAS_BAD_CREDIT"] > 0)
    ).astype(int)

    grp = temp_df.groupby("SK_ID_CURR")

    custom_aggs = {

        "HAS_ACTIVE_CREDIT": "sum",
        "HAS_CLOSED_CREDIT": "sum",
        "HAS_BAD_CREDIT": "sum",
        "HAS_ANY_OVERDUE_DEBT": "max",
        "HAS_SIGNIFICANT_OVERDUE_DEBT": "max",

        "HAS_ANY_MAJOR_BUREAU_RISK": "max",
        "CREDIT_UTILIZATION": ["mean", "max"],
        "YEARS_CREDIT_UPDATE": [
            "min",
            "max",
        ],
        "CREDIT_ENDDATE_PROXIMITY": "mean",
    }

    custom_agg_df = grp.agg(custom_aggs)
    custom_agg_df.columns = [
        f"BUREAU_{col[0]}_{col[1]}" for col in custom_agg_df.columns
    ]

    count_features = pd.DataFrame()
    count_features["BUREAU_CREDIT_COUNT"] = grp.size() 
    count_features["BUREAU_ACTIVE_CREDIT_COUNT"] = grp[
        "HAS_ACTIVE_CREDIT"
    ].sum()
    count_features = count_features.reset_index()

    custom_agg_df = custom_agg_df.merge(count_features, on="SK_ID_CURR", how="left")

    return custom_agg_df


def create_cc_balance_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates human-engineered features from credit card balance data.
    These features capture the client's credit card usage, payment behavior,
    delinquency, and rolling credit utilization, aggregated by customer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing credit card balance records with at least the following columns:
        - 'SK_ID_CURR': customer ID
        - 'MONTHS_BALANCE': months since the start of observation
        - 'AMT_BALANCE': current balance
        - 'AMT_CREDIT_LIMIT_ACTUAL': credit limit
        - 'AMT_PAYMENT_CURRENT': current payment amount
        - 'AMT_INST_MIN_REGULARITY': minimum required installment
        - 'AMT_DRAWINGS_ATM_CURRENT': ATM cash drawings
        - 'AMT_DRAWINGS_CURRENT': total drawings
        - 'SK_DPD': days past due
        - 'SK_DPD_DEF': days past due for defaulted payments
        - 'NAME_CONTRACT_STATUS': status of the credit contract
        - 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE'

    Returns
    -------
    pd.DataFrame
        Aggregated customer-level features with columns like:
        - CC_CREDIT_UTILIZATION_RATIO_mean, CC_CREDIT_UTILIZATION_RATIO_max, CC_CREDIT_UTILIZATION_RATIO_last
        - CC_AMT_BALANCE_mean, CC_AMT_BALANCE_max, CC_AMT_BALANCE_sum, CC_AMT_BALANCE_std
        - CC_AMT_PAYMENT_CURRENT_mean, CC_AMT_PAYMENT_CURRENT_sum, CC_AMT_PAYMENT_CURRENT_std
        - CC_MIN_PAYMENT_RATIO_mean, CC_MIN_PAYMENT_RATIO_min
        - CC_MADE_MINIMUM_PAYMENT_mean
        - CC_UTILIZATION_ROLLING_MEAN_first, CC_UTILIZATION_ROLLING_MEAN_last
        - CC_IS_DELINQUENT_max, CC_IS_SERIOUSLY_DELINQUENT_max
        - and other ratios capturing payment vs balance, drawings, and activity

    Notes
    -----
    - Negative balances or receivables are flagged and replaced with 0
    - 'CREDIT_UTILIZATION_RATIO' = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL, clipped to 1.5
    - 'MIN_PAYMENT_RATIO' = AMT_PAYMENT_CURRENT / AMT_INST_MIN_REGULARITY
    - 'MADE_MINIMUM_PAYMENT' = 1 if MIN_PAYMENT_RATIO >= 0.95
    - 'ATM_DRAWING_RATIO' = AMT_DRAWINGS_ATM_CURRENT / AMT_DRAWINGS_CURRENT
    - 'IS_DELINQUENT' = 1 if SK_DPD > 0
    - 'IS_SERIOUSLY_DELINQUENT' = 1 if SK_DPD > 30
    - 'UTILIZATION_ROLLING_MEAN' = rolling 3-month mean of CREDIT_UTILIZATION_RATIO per customer
    - 'IS_ACTIVE' = 1 if NAME_CONTRACT_STATUS == "Active"
    - 'DRAWINGS_TO_PAYMENTS_RATIO' = AMT_DRAWINGS_CURRENT / (AMT_PAYMENT_CURRENT + 1)
    - 'PAYMENT_TO_BALANCE_RATIO' = AMT_PAYMENT_CURRENT / (AMT_BALANCE + 1)
    """
    temp_df = df.copy()

    temp_df["NEGATIVE_AMT_BALANCE"] = (temp_df["AMT_BALANCE"] < 0).astype(int)
    temp_df["AMT_BALANCE"] = temp_df["AMT_BALANCE"].clip(0)
    temp_df["CREDIT_UTILIZATION_RATIO"] = (
        temp_df["AMT_BALANCE"] / temp_df["AMT_CREDIT_LIMIT_ACTUAL"]
    )

    temp_df["CREDIT_UTILIZATION_RATIO"] = np.where(
        temp_df["AMT_CREDIT_LIMIT_ACTUAL"] > 0,
        temp_df["CREDIT_UTILIZATION_RATIO"].clip(
            0, 1.5
        ), 
        0,
    )

    temp_df["MIN_PAYMENT_RATIO"] = (
        temp_df["AMT_PAYMENT_CURRENT"] / temp_df["AMT_INST_MIN_REGULARITY"]
    )

    temp_df["MIN_PAYMENT_RATIO"] = np.where(
        temp_df["AMT_INST_MIN_REGULARITY"] > 0, temp_df["MIN_PAYMENT_RATIO"], 1.0
    )
    temp_df["MADE_MINIMUM_PAYMENT"] = (temp_df["MIN_PAYMENT_RATIO"] >= 0.95).astype(
        int
    ) 

    temp_df["AMT_DRAWINGS_ATM_CURRENT"] = temp_df["AMT_DRAWINGS_ATM_CURRENT"].clip(0)
    temp_df["AMT_DRAWINGS_CURRENT"] = temp_df["AMT_DRAWINGS_CURRENT"].clip(0)
    temp_df["ATM_DRAWING_RATIO"] = np.where(
        temp_df["AMT_DRAWINGS_CURRENT"] > 0,
        temp_df["AMT_DRAWINGS_ATM_CURRENT"] / temp_df["AMT_DRAWINGS_CURRENT"],
        0,
    )

    temp_df["IS_DELINQUENT"] = (temp_df["SK_DPD"] > 0).astype(int)
    temp_df["IS_SERIOUSLY_DELINQUENT"] = (temp_df["SK_DPD"] > 30).astype(int)

    temp_df["NEGATIVE_AMT_RECEIVABLE_PRINCIPAL"] = (
        temp_df["AMT_RECEIVABLE_PRINCIPAL"] < 0
    ).astype(int)
    temp_df["AMT_RECEIVABLE_PRINCIPAL"] = temp_df["AMT_RECEIVABLE_PRINCIPAL"].clip(0)

    temp_df["NEGATIVE_AMT_RECIVABLE"] = (temp_df["AMT_RECIVABLE"] < 0).astype(int)
    temp_df["AMT_RECIVABLE"] = temp_df["AMT_RECIVABLE"].clip(0)

    temp_df["NEGATIVE_AMT_TOTAL_RECEIVABLE"] = (
        temp_df["AMT_TOTAL_RECEIVABLE"] < 0
    ).astype(int)
    temp_df["AMT_TOTAL_RECEIVABLE"] = temp_df["AMT_TOTAL_RECEIVABLE"].clip(0)

    temp_df["SK_DPD"] = temp_df["SK_DPD"].clip(0, 365)
    temp_df["SK_DPD_DEF"] = temp_df["SK_DPD_DEF"].clip(0, 365)

    temp_df = temp_df.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"])

    temp_df["UTILIZATION_ROLLING_MEAN"] = (
        temp_df.groupby("SK_ID_CURR")["CREDIT_UTILIZATION_RATIO"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )

    temp_df["IS_ACTIVE"] = (temp_df["NAME_CONTRACT_STATUS"] == "Active").astype(int)

    temp_df["DRAWINGS_TO_PAYMENTS_RATIO"] = temp_df["AMT_DRAWINGS_CURRENT"] / (
        temp_df["AMT_PAYMENT_CURRENT"] + 1
    )
    temp_df["PAYMENT_TO_BALANCE_RATIO"] = temp_df["AMT_PAYMENT_CURRENT"] / (
        temp_df["AMT_BALANCE"] + 1
    )


    grp = temp_df.groupby("SK_ID_CURR")

    custom_aggs = {
        "CREDIT_UTILIZATION_RATIO": ["mean", "max", "last"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max", "last"],
        "AMT_BALANCE": ["mean", "max", "sum", "std"],
        "AMT_PAYMENT_CURRENT": ["mean", "sum", "std"],
        "MIN_PAYMENT_RATIO": ["mean", "min"], 
        "MADE_MINIMUM_PAYMENT": "mean",
        "AMT_DRAWINGS_CURRENT": ["mean", "max", "sum"],
        "ATM_DRAWING_RATIO": "mean",
        "CNT_DRAWINGS_CURRENT": ["mean", "max", "sum"],
        "UTILIZATION_ROLLING_MEAN": ["first", "last"],
        "DRAWINGS_TO_PAYMENTS_RATIO": ["mean", "max"],
        "PAYMENT_TO_BALANCE_RATIO": ["mean", "max"],
        "IS_ACTIVE": ["mean"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "IS_DELINQUENT": "max", 
        "IS_SERIOUSLY_DELINQUENT": "max",
    }

    custom_agg_df = grp.agg(custom_aggs)
    custom_agg_df.columns = [f"CC_{col[0]}_{col[1]}" for col in custom_agg_df.columns]
    return custom_agg_df.reset_index()


def create_pos_cash_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates human-engineered features from POS/CASH balance data.
    These features capture payment behavior, delinquency, and contract status
    for installment loans, aggregated by customer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing POS/CASH records with at least the following columns:
        - 'SK_ID_CURR': customer ID
        - 'MONTHS_BALANCE': months since the start of observation
        - 'SK_DPD': days past due
        - 'SK_DPD_DEF': days past due for defaulted payments
        - 'NAME_CONTRACT_STATUS': status of the contract
        - 'CNT_INSTALMENT': total number of installments
        - 'CNT_INSTALMENT_FUTURE': number of remaining installments

    Returns
    -------
    pd.DataFrame
        Aggregated customer-level features with columns like:
        - POS_SK_DPD_max, POS_SK_DPD_mean, POS_SK_DPD_last, POS_SK_DPD_std
        - POS_SK_DPD_DEF_max, POS_SK_DPD_DEF_mean
        - POS_IS_DELINQUENT_max, POS_IS_SERIOUSLY_DELINQUENT_max, POS_IS_DELINQUENT_DEF_max
        - POS_IS_ACTIVE_mean, POS_IS_COMPLETED_max, POS_IS_OTHER_STATUS_max
        - POS_INSTALMENTS_COMPLETED_RATIO_mean, POS_INSTALMENTS_COMPLETED_RATIO_max
        - POS_DPD_CHANGE_max

    Notes
    -----
    - 'SK_DPD' and 'SK_DPD_DEF' are clipped between 0 and 365
    - 'IS_DELINQUENT' = 1 if SK_DPD > 0
    - 'IS_SERIOUSLY_DELINQUENT' = 1 if SK_DPD > 30
    - 'IS_DELINQUENT_DEF' = 1 if SK_DPD_DEF > 0
    - 'IS_ACTIVE', 'IS_COMPLETED', 'IS_OTHER_STATUS' are derived from NAME_CONTRACT_STATUS
    - 'INSTALMENTS_COMPLETED_RATIO' = 1 - CNT_INSTALMENT_FUTURE / (CNT_INSTALMENT + 1)
    - 'DPD_CHANGE' = change in SK_DPD from first to last observation per customer
    """
    temp_df = df.copy()

    temp_df = temp_df.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"])

    temp_df["SK_DPD"] = temp_df["SK_DPD"].clip(0, 365)
    temp_df["SK_DPD_DEF"] = temp_df["SK_DPD_DEF"].clip(0, 365)
    temp_df["IS_DELINQUENT"] = (temp_df["SK_DPD"] > 0).astype(int)
    temp_df["IS_SERIOUSLY_DELINQUENT"] = (temp_df["SK_DPD"] > 30).astype(int)
    temp_df["IS_DELINQUENT_DEF"] = (temp_df["SK_DPD_DEF"] > 0).astype(int)

    temp_df["DPD_CHANGE"] = temp_df.groupby("SK_ID_CURR")["SK_DPD"].transform(
        lambda x: x.iloc[-1] - x.iloc[0]
    )

    temp_df["IS_ACTIVE"] = (temp_df["NAME_CONTRACT_STATUS"] == "Active").astype(int)
    temp_df["IS_COMPLETED"] = (temp_df["NAME_CONTRACT_STATUS"] == "Completed").astype(
        int
    )

    temp_df["IS_OTHER_STATUS"] = (
        temp_df["NAME_CONTRACT_STATUS"]
        .isin(
            [
                "Signed",
                "Demand",
                "Returned to the store",
                "Approved",
                "Amortized debt",
                "Canceled",
            ]
        )
        .astype(int)
    )

    temp_df["INSTALMENTS_COMPLETED_RATIO"] = 1 - (
        temp_df["CNT_INSTALMENT_FUTURE"] / (temp_df["CNT_INSTALMENT"] + 1)
    )

    grp = temp_df.groupby("SK_ID_CURR")

    custom_aggs = {
        "SK_DPD": ["max", "mean", "last", "std"],
        "SK_DPD_DEF": ["max", "mean"],
        "IS_DELINQUENT": "max",
        "IS_SERIOUSLY_DELINQUENT": "max",
        "IS_DELINQUENT_DEF": "max",
        "IS_ACTIVE": "mean",
        "IS_COMPLETED": "max",
        "IS_OTHER_STATUS": "max",
        "INSTALMENTS_COMPLETED_RATIO": [
            "mean",
            "max",
        ],
        "DPD_CHANGE": "max",
    }

    custom_agg_df = grp.agg(custom_aggs)
    custom_agg_df.columns = [f"POS_{col[0]}_{col[1]}" for col in custom_agg_df.columns]
    return custom_agg_df.reset_index()


def create_prev_app_custom_features_balanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates human-engineered features from previous applications data,
    capturing approval/refusal patterns and credit ratios while keeping
    computation efficient (balanced version).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of previous applications, must contain at least the following columns:
        - 'SK_ID_CURR': customer ID
        - 'NAME_CONTRACT_STATUS': contract status ('Approved', 'Refused', 'Canceled')
        - 'AMT_APPLICATION': requested loan amount
        - 'AMT_CREDIT': granted credit amount
        - 'AMT_ANNUITY': annuity amount
        - 'AMT_DOWN_PAYMENT': down payment amount
        - 'AMT_GOODS_PRICE': goods price
        - 'CNT_PAYMENT': number of payments
        - 'YEARS_DECISION': years since decision
        - 'NAME_CONTRACT_TYPE': contract type

    Returns
    -------
    pd.DataFrame
        Aggregated customer-level features with columns including:
        - PREV_WAS_APPROVED_mean, PREV_WAS_APPROVED_sum
        - PREV_WAS_REFUSED_mean, PREV_WAS_REFUSED_sum
        - PREV_WAS_CANCELED_mean, PREV_WAS_CANCELED_sum
        - PREV_APP_CREDIT_DIFF_mean, PREV_APP_CREDIT_DIFF_max
        - PREV_APP_CREDIT_RATIO_mean, PREV_APP_CREDIT_RATIO_min, PREV_APP_CREDIT_RATIO_max
        - PREV_AMT_APPLICATION_mean, PREV_AMT_APPLICATION_max
        - PREV_AMT_CREDIT_mean, PREV_AMT_CREDIT_max
        - PREV_AMT_ANNUITY_mean, PREV_AMT_ANNUITY_max
        - PREV_AMT_DOWN_PAYMENT_mean, PREV_AMT_DOWN_PAYMENT_max
        - PREV_AMT_GOODS_PRICE_mean, PREV_AMT_GOODS_PRICE_max
        - PREV_CNT_PAYMENT_mean, PREV_CNT_PAYMENT_max
        - PREV_YEARS_DECISION_min, PREV_YEARS_DECISION_max, PREV_YEARS_DECISION_mean
        - PREV_NAME_CONTRACT_TYPE_PRODUCT_DIVERSITY
        - PREV_APP_COUNT, PREV_APP_APPROVED_COUNT, PREV_APP_REFUSED_COUNT
        - PREV_APPROVAL_RATE, PREV_REFUSAL_RATE, PREV_REFUSAL_TO_APPROVAL

    Notes
    -----
    - Flags are created for contract status: WAS_APPROVED, WAS_REFUSED, WAS_CANCELED
    - APP_CREDIT_DIFF = AMT_APPLICATION - AMT_CREDIT
    - APP_CREDIT_RATIO = AMT_CREDIT / AMT_APPLICATION, clipped to [0, 5]
    - Aggregation is done per customer (SK_ID_CURR)
    - Approval/refusal rates are computed with division by at least 1 to avoid zero division
    """
    temp_df = df.copy()

    temp_df["WAS_APPROVED"] = (temp_df["NAME_CONTRACT_STATUS"] == "Approved").astype(
        int
    )
    temp_df["WAS_REFUSED"] = (temp_df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    temp_df["WAS_CANCELED"] = (temp_df["NAME_CONTRACT_STATUS"] == "Canceled").astype(
        int
    )

    temp_df["APP_CREDIT_DIFF"] = temp_df["AMT_APPLICATION"] - temp_df["AMT_CREDIT"]
    temp_df["APP_CREDIT_RATIO"] = (
        (temp_df["AMT_CREDIT"] / temp_df["AMT_APPLICATION"])
        .replace([np.inf, -np.inf], np.nan)
        .clip(0, 5)
    )

    grp = temp_df.groupby("SK_ID_CURR")

    custom_aggs = {
        "WAS_APPROVED": ["mean", "sum"],
        "WAS_REFUSED": ["mean", "sum"],
        "WAS_CANCELED": ["mean", "sum"],
        "APP_CREDIT_DIFF": ["mean", "max"],
        "APP_CREDIT_RATIO": ["mean", "min", "max"],
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max"],
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_DOWN_PAYMENT": ["mean", "max"],
        "AMT_GOODS_PRICE": ["mean", "max"],
        "CNT_PAYMENT": ["mean", "max"],
        "YEARS_DECISION": ["min", "max", "mean"],
        "NAME_CONTRACT_TYPE": [("PRODUCT_DIVERSITY", "nunique")],
    }

    agg_df = grp.agg(custom_aggs)
    agg_df.columns = [f"PREV_{c[0]}_{c[1]}" for c in agg_df.columns]
    agg_df = agg_df.reset_index()

    count_df = pd.DataFrame(
        {
            "SK_ID_CURR": grp.size().index,
            "PREV_APP_COUNT": grp.size().values,
            "PREV_APP_APPROVED_COUNT": grp["WAS_APPROVED"].sum().values,
            "PREV_APP_REFUSED_COUNT": grp["WAS_REFUSED"].sum().values,
        }
    )
    agg_df = agg_df.merge(count_df, on="SK_ID_CURR", how="left")

    agg_df["PREV_APPROVAL_RATE"] = agg_df["PREV_APP_APPROVED_COUNT"] / agg_df[
        "PREV_APP_COUNT"
    ].replace(0, 1)
    agg_df["PREV_REFUSAL_RATE"] = agg_df["PREV_APP_REFUSED_COUNT"] / agg_df[
        "PREV_APP_COUNT"
    ].replace(0, 1)
    agg_df["PREV_REFUSAL_TO_APPROVAL"] = agg_df["PREV_APP_REFUSED_COUNT"] / agg_df[
        "PREV_APP_APPROVED_COUNT"
    ].replace(0, 1)

    return agg_df


def create_bureau_balance_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create custom aggregated features from the `bureau_balance` dataset.

    This function engineers features that summarize a client’s credit history 
    from monthly bureau balance records. It maps raw status codes into 
    severity levels, flags severe delinquency and write-offs, and computes 
    aggregated statistics per credit record (`SK_ID_BUREAU`).

    Parameters
    ----------
    df : pd.DataFrame
        Input bureau_balance dataframe with at least the following columns:
        - "SK_ID_BUREAU": unique credit record identifier
        - "STATUS": credit status (categorical codes: "C", "X", "0", "1", ..., "5")
        - "MONTHS_BALANCE": time index of bureau balance record

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with one row per `SK_ID_BUREAU` containing:
        - `BB_STATUS_SEVERITY_max`: maximum severity across history
        - `BB_STATUS_SEVERITY_mean`: mean severity across history
        - `BB_STATUS_SEVERITY_last`: last observed severity
        - `BB_WAS_SEVERELY_DELINQUENT_max`: whether any delinquency >=3 occurred
        - `BB_WAS_WRITTEN_OFF_max`: whether any write-off occurred
        - `BB_MONTHS_BALANCE_count`: number of months observed
    """

    temp_df = df.copy()

    status_map = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

    temp_df["STATUS_SEVERITY"] = temp_df["STATUS"].map(status_map)

    temp_df["WAS_SEVERELY_DELINQUENT"] = (temp_df["STATUS_SEVERITY"] >= 3).astype(int)
    temp_df["WAS_WRITTEN_OFF"] = (temp_df["STATUS"] == "5").astype(int)

    grp_credit = temp_df.groupby("SK_ID_BUREAU")

    credit_aggs = {
        "STATUS_SEVERITY": ["max", "mean", "last"],
        "WAS_SEVERELY_DELINQUENT": "max",
        "WAS_WRITTEN_OFF": "max",
        "MONTHS_BALANCE": "count",
    }

    credit_agg_df = grp_credit.agg(credit_aggs)
    credit_agg_df.columns = [f"BB_{col[0]}_{col[1]}" for col in credit_agg_df.columns]
    credit_agg_df = credit_agg_df.reset_index()

    return credit_agg_df


def engineer_features_from_table(
    auxiliary_df: pd.DataFrame,
    group_id: str,
    num_aggs: Dict[str, List[str]] = None,
    cat_aggs: Dict[str, List[str]] = None,
    custom_funcs: List[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    fillna: object = 0,
) -> pd.DataFrame:
    """
    Aggregates auxiliary_df to group_id level using num_aggs + cat_aggs (if provided),
    applies custom_funcs (each should return a DataFrame indexed by group_id),
    merges results and returns the full aggregated DataFrame for deployment.

    IMPORTANT: This function DOES NOT run any feature selection. It just returns
    the full set of aggregated + custom features (group_id column included).

    Parameters
    ----------
    auxiliary_df : pd.DataFrame
        Raw auxiliary table (e.g. previous_application, credit_card_balance).
    group_id : str
        Column name used to group (e.g. "SK_ID_CURR" or "SK_ID_BUREAU").
    num_aggs : dict, optional
        Numeric aggregations mapping {col_name: [agg1, agg2, ...]}.
        Use the exact dicts you used in training (e.g. cc_num_aggs, prev_num_aggs).
    cat_aggs : dict, optional
        Categorical aggregations mapping {col_name: [agg1, ...]}.
    custom_funcs : list of callables, optional
        Functions that accept the raw auxiliary_df and return a DataFrame
        aggregated to group_id (must include group_id column).
    fillna : scalar or dict-like
        Value to fill NA after merging (default 0). If None, no fillna applied.

    Returns
    -------
    agg_df : pd.DataFrame
        Aggregated and merged features at group_id level (group_id column first).
    """
    num_aggs = {} if num_aggs is None else num_aggs
    cat_aggs = {} if cat_aggs is None else cat_aggs
    custom_funcs = [] if custom_funcs is None else custom_funcs

    agg_dict = {**num_aggs, **cat_aggs}
    if len(agg_dict) > 0:
        grp = auxiliary_df.groupby(group_id)
        agg_df = grp.agg(agg_dict)
        agg_df.columns = [f"{col[0]}_{col[1]}" for col in agg_df.columns]
        agg_df = agg_df.reset_index()
    else:
        if group_id in auxiliary_df.columns:
            agg_df = auxiliary_df[[group_id]].drop_duplicates().reset_index(drop=True)
        else:
            raise ValueError(f"group_id '{group_id}' not found in auxiliary_df")

    for func in custom_funcs:
        custom_feats = func(auxiliary_df)
        if group_id not in custom_feats.columns:
            raise ValueError(
                f"Custom function {func.__name__} must return a DataFrame containing '{group_id}' column"
            )
        agg_df = agg_df.merge(custom_feats, on=group_id, how="left")

    if fillna is not None:
        agg_df = agg_df.fillna(fillna)

    return agg_df


ins_num_aggs = {
    "AMT_INSTALMENT": ["mean", "max", "sum"],
    "AMT_PAYMENT": ["mean", "max", "sum"],
    "DAYS_INSTALMENT": ["mean", "max"],
    "DAYS_ENTRY_PAYMENT": ["mean", "max"],
}


bureau_num_aggs = {
    "AMT_CREDIT_SUM": ["mean", "max", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "max", "sum"],
    "AMT_CREDIT_SUM_OVERDUE": ["sum"],
    "YEARS_CREDIT": ["min", "max", "mean"],
    "YEARS_CREDIT_ENDDATE": ["min", "max", "mean"],
    "YEARS_CREDIT_UPDATE": [
        "min",
        "max",
        "mean",
    ],
}


bb_num_aggs = {
    "BB_STATUS_SEVERITY_max": ["mean", "max"],
    "BB_STATUS_SEVERITY_last": ["mean", "max"],
    "BB_WAS_SEVERELY_DELINQUENT_max": "max",
    "BB_WAS_WRITTEN_OFF_max": "max",
    "BB_MONTHS_BALANCE_count": [
        "mean",
        "sum",
    ],
}


cc_num_aggs = {
    "MONTHS_BALANCE": ["min", "max", "size"],
    "AMT_DRAWINGS_ATM_CURRENT": ["sum", "max", "mean"],
    "AMT_DRAWINGS_OTHER_CURRENT": ["sum", "max", "mean"],
    "AMT_DRAWINGS_POS_CURRENT": ["sum", "max", "mean"],
    "AMT_PAYMENT_CURRENT": ["min", "sum", "mean"],
    "AMT_PAYMENT_TOTAL_CURRENT": ["min", "max", "sum", "mean"],
    "AMT_RECIVABLE": ["mean", "max", "sum"],
    "AMT_RECEIVABLE_PRINCIPAL": ["mean", "max"],
    "AMT_TOTAL_RECEIVABLE": ["mean", "max", "sum"],
    "CNT_DRAWINGS_ATM_CURRENT": ["sum", "max", "mean"],
    "CNT_DRAWINGS_OTHER_CURRENT": ["sum", "mean"],
    "CNT_DRAWINGS_POS_CURRENT": ["sum", "max", "mean"],
    "CNT_INSTALMENT_MATURE_CUM": ["mean", "max"],
}


prev_num_aggs = {
    "AMT_APPLICATION": ["mean", "max"],
    "AMT_CREDIT": ["mean", "max"],
    "AMT_ANNUITY": ["mean", "max"],
    "AMT_DOWN_PAYMENT": ["mean", "max"],
    "AMT_GOODS_PRICE": ["mean", "max"],
    "CNT_PAYMENT": ["mean", "max"],
    "YEARS_DECISION": ["min", "max", "mean"],
}


pos_num_aggs = {
    "CNT_INSTALMENT": ["mean", "max"], 
    "CNT_INSTALMENT_FUTURE": ["mean", "min", "max"],
    "MONTHS_BALANCE": ["min", "max", "mean"],
}