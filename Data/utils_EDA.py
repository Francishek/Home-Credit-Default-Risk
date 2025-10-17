import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional, List, Tuple, Union



def feature_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Print numeric, categorical and binary features of dataframe.
    """
    categorical = data.select_dtypes(include="object").columns.tolist()
    numerical = data.select_dtypes(exclude="object").columns.tolist()
    binary = [col for col in data.columns if data[col].nunique() == 2]

    print(f"Numerical features: {numerical}")
    print(f"Categorical features: {categorical}")
    print(f"Binary features: {binary}")


def missing_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe that contains missing column names and
    percent of missing values in relation to the whole dataframe.

    dataframe: dataframe that gives the column names and their % of missing values
    """

    missing_values = dataframe.isnull().sum().sort_values(ascending=False)

    missing_values_pct = 100 * missing_values / len(dataframe)

    concat_values = pd.concat(
        [missing_values, missing_values / len(dataframe), missing_values_pct.round(1)],
        axis=1,
    )

    concat_values.columns = ["Missing Count", "Missing Count Ratio", "Missing Count %"]

    return concat_values[concat_values.iloc[:, 1] != 0]


def calculate_missing_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prints number of rows with missing values.
    Prints percentage of rows with missing values in dataset.
    """
    df_no_Na = data.dropna()
    missing_rows = data.shape[0] - df_no_Na.shape[0]
    percentage_missing = missing_rows / data.shape[0] * 100
    print(
        f"Missing rows: {missing_rows} of {data.shape[0]} total rows in data set."
        f"\nMissing rows %: {percentage_missing:.2f}"
    )


def plot_target_distribution(df: pd.DataFrame) -> None:
    """
    Generates and displays a pie chart showing the distribution of the
    'TARGET' status in a DataFrame with user-friendly labels.
    """
    target_counts = df["TARGET"].value_counts().sort_index()

    # Create descriptive labels for the pie chart
    descriptive_labels = ["Repaid Loan", "Defaulted"]

    sizes = target_counts.values

    def autopct_with_counts(pct: float) -> str:
        total = sum(sizes)
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count})"

    plt.figure(figsize=(5, 5))
    plt.pie(
        sizes,
        labels=descriptive_labels,  # Use the descriptive labels
        autopct=autopct_with_counts,
        startangle=90,
        textprops={"fontsize": 12},
    )
    plt.title("TARGET Status Distribution: 0 = Repaid Loan, 1 = Defaulted", fontsize=14)
    plt.axis("equal")
    plt.show()


def classify_features(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    discrete_threshold: int = 20,
    cat_threshold: int = 10,
):
    """
    Classify features into groups:
    - continuous numeric
    - discrete numeric
    - low-card categorical
    - medium/high-card categorical
    - date features
    """

    results = []

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    for col in num_cols:
        nunique = df[col].nunique()

        if col.startswith(("DAYS_", "YEARS_")):
            feature_type = "date"
        elif nunique < discrete_threshold:
            feature_type = "discrete_numeric"
        else:
            feature_type = "continuous_numeric"

        results.append((col, nunique, feature_type))

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in cat_cols:
        nunique = df[col].nunique()

        if nunique <= cat_threshold:
            feature_type = "low_cardinality_categorical"
        else:
            feature_type = "medium_high_cardinality_categorical"

        results.append((col, nunique, feature_type))

    summary = pd.DataFrame(results, columns=["feature", "nunique", "classification"])

    return summary


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


def impute_missing_values(
    df: pd.DataFrame, 
    feature_summary: pd.DataFrame, 
    target: str = "TARGET"
) -> pd.DataFrame:
    """
    Impute missing values according to feature classification.

    Rules:
    - Continuous numeric: median
    - Discrete numeric: mode
    - Categorical: "Unknown"
    - Dates: median

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with missing values
    feature_summary : pd.DataFrame
        DataFrame with columns ["feature", "classification"]
    target : str
        Name of the target column (ignored in imputation)

    Returns:
    --------
    df_imputed : pd.DataFrame
        Copy of df with missing values imputed
    """
    df = df.copy()

    for _, row in feature_summary.iterrows():
        col = row["feature"]
        if col == target or col not in df.columns:
            continue

        classification = row["classification"]

        if classification == "continuous_numeric":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        elif classification == "discrete_numeric":
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                mode_val = mode_val[0]
            else:
                mode_val = 0  # fallback
            df[col] = df[col].fillna(mode_val)

        elif classification in [
            "low_cardinality_categorical",
            "medium_high_cardinality_categorical",
        ]:
            df[col] = df[col].fillna("Unknown")

        elif classification == "date":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    return df


def plot_distribution_numerical(
    data: pd.DataFrame,
    column: str,
    target: str = "TARGET",
    figsize: Tuple[int, int] = (8, 6),
    bins: int = 10,
    log_scale: bool = False,
) -> None:
    """
    Plot comparative analysis of a numerical feature vs target variable
    including general and conditional distributions (kdeplot) and boxplots.

    Parameters:
    - data: Input DataFrame
    - column: Numerical column to analyze
    - target: Target variable name (default: Transported)
    - figsize: Figure size (width, height)
    - bins: Number of histogram bins (default: 10)
    - log_scale: Whether to apply log1p scale to the column (default: False)
    """
    col_data = np.log1p(data[column]) if log_scale else data[column]
    plot_data = data.copy()
    plot_data["_col_"] = col_data

    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    sns.boxplot(
        y=plot_data["_col_"],
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    plt.title(f"General Boxplot of {column}" + (" (log scale)" if log_scale else ""))
    plt.xlabel("")
    plt.ylabel(column)

    plt.subplot(2, 2, 2)
    sns.boxplot(
        x=target,
        y="_col_",
        data=plot_data,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    plt.title(
        f"{column.capitalize()} by {target}" + (" (log scale)" if log_scale else "")
    )
    plt.xlabel(target)
    plt.ylabel(column)

    plt.subplot(2, 2, 3)
    sns.histplot(plot_data["_col_"].dropna(), bins=bins, kde=True, color="skyblue")
    plt.title(f"Distribution of {column}" + (" (log scale)" if log_scale else ""))
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)

    sns.kdeplot(data=plot_data, x="_col_", hue=target, common_norm=False)
    plt.title(f"{column.capitalize()} distribution by {target}")
    plt.xlabel(column)

    plt.tight_layout()
    plt.show()


def stacked_bar_with_percent(
    data: pd.DataFrame,
    column_x: str,
    column_y: str = "TARGET",
    figsize: Tuple[int, int] = (8, 4),
    top_n: int = None,
    orientation: str = "vertical",
    sort_by_default: bool = False, 
    ascending: bool = True, 
) -> None:
    """
    Plot a stacked bar chart with bars sized by actual frequency and
    annotated with percentages for binary target analysis.
    Options:
      • plot only top N categories,
      • change orientation,
      • sort categories by default risk (ascending or descending).
    """

    if top_n is not None:
        top_categories = data[column_x].value_counts().nlargest(top_n).index
        data = data[data[column_x].isin(top_categories)]

    if sort_by_default:
        risk_order = (
            data.groupby(column_x, observed=False)[column_y]
            .mean() 
            .sort_values(ascending=ascending) 
            .index
        )
        data = data.copy()
        data[column_x] = pd.Categorical(
            data[column_x], categories=risk_order, ordered=True
        )

    count_table = pd.crosstab(data[column_x], data[column_y])

    if orientation == "horizontal" and sort_by_default and not ascending:
        count_table = count_table.iloc[::-1]

    percent_table = count_table.div(count_table.sum(axis=1), axis=0) * 100

    plot_kind = "bar" if orientation == "vertical" else "barh"
    ax = count_table.plot(kind=plot_kind, stacked=True, figsize=figsize)

    for i, category in enumerate(count_table.index):
        bottom = 0
        for stroke_value in count_table.columns:
            count = count_table.loc[category, stroke_value]
            percent = percent_table.loc[category, stroke_value]
            if count > 0:
                if orientation == "vertical":
                    ax.text(
                        i,
                        bottom + count / 2,
                        f"{percent:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                        bbox=dict(
                            facecolor="white", alpha=0.7, edgecolor="none", pad=2
                        ),
                    )
                else: 
                    ax.text(
                        bottom + count / 2,
                        i,
                        f"{percent:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                        bbox=dict(
                            facecolor="white", alpha=0.7, edgecolor="none", pad=2
                        ),
                    )
            bottom += count

    if orientation == "vertical":
        plt.xlabel(column_x)
        plt.ylabel("Number of persons")
        plt.xticks(rotation=0, ha="right")
    else:
        plt.xlabel("Number of persons")
        plt.ylabel(column_x)
        plt.xticks(rotation=0)

    plt.title(f"Defaulted by {column_x}", pad=15)
    plt.legend(title="Defaulted", labels=["No", "Yes"], loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()


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
