import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def assign_phase(month: str) -> str:
    """
    Assigns a descriptive phase name based on the month.
    (From ya.ipynb)

    Args:
        month (str): The month in 'YYYY-MM' format.

    Returns:
        str: The phase name.
    """
    if month < "2022-10":
        return "Pre-Rating Impact (before Oct. 2022)"
    elif "2022-10" <= month <= "2022-12":
        return "Transition (Oct. -Dec. 2022)"
    elif "2023-01" <= month <= "2023-02":
        return "Rating-impact for all (Jan. - Feb. 2023)"
    else:
        return "Stabilized (Feb.-Jun. 2023)"

def gini_coefficient(x: np.ndarray) -> float:
    """
    Computes the Gini coefficient of a numpy array.
    (From ya.ipynb)

    Args:
        x (np.ndarray): The input array.

    Returns:
        float: The Gini coefficient.
    """
    x = np.sort(np.abs(x))  # Use absolute values
    n = len(x)
    if n == 0 or np.sum(x) == 0:
        return np.nan
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def bimodality_coefficient(data: np.ndarray) -> float:
    """
    Calculates the bimodality coefficient for a given dataset.
    (From ya.ipynb)

    Args:
        data (np.ndarray): The input data.

    Returns:
        float: The bimodality coefficient.
    """
    n = len(data)
    if n == 0:
        return np.nan
    g = skew(data)
    k = kurtosis(data, fisher=False) # Use Pearson's definition of kurtosis
    if k == 0:
        return np.nan
    return (g**2 + 1) / k



def add_cohort_and_period(
    df,
    id_col="raterParticipantId",
    date_col="week_dt",
    legacy_cutoff="2022-10-01",
    new_end="2023-01-01",
    change_point="2023-01-01",
):
    """
    Adds 'Cohort' and 'Period' columns to the dataframe.

    Cohort:
      - 'Legacy Users' if first_seen < legacy_cutoff
      - 'New Users'    if legacy_cutoff <= first_seen < new_end
      - 'Other'        otherwise (excluded from plot)

    Period:
      - 'Before' if week_dt < change_point
      - 'After'  otherwise
    """
    # Ensure datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    legacy_cutoff = pd.to_datetime(legacy_cutoff)
    new_end = pd.to_datetime(new_end)
    change_point = pd.to_datetime(change_point)

    # First appearance per user
    first_seen = (
        df.groupby(id_col)[date_col].min().rename("first_seen").reset_index()
    )
    df = df.merge(first_seen, on=id_col, how="left")

    # Assign cohort
    conditions = [
        df["first_seen"] < legacy_cutoff,
        (df["first_seen"] >= legacy_cutoff) & (df["first_seen"] < new_end),
    ]
    choices = ["Legacy Users", "New Users"]
    df["Cohort"] = np.select(conditions, choices, default="Other")

    # Assign period
    df["Period"] = np.where(df[date_col] < change_point, "Before", "After")

    return df


def add_phase(df, date_col="week_dt"):
    """
    Adds a 'Phase' column based on week_dt:
      Oct → Jan : 2022-10-01 <= week_dt < 2023-01-01
      Jan → Apr : 2023-01-01 <= week_dt < 2023-04-01
      Apr → Jun : 2023-04-01 <= week_dt < 2023-07-01
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    bins = [
        pd.Timestamp("2022-10-01"),
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-04-01"),
        pd.Timestamp("2023-07-01"),
    ]
    labels = ["Oct → Jan", "Jan → Apr", "Apr → Jun"]

    df["Phase"] = pd.cut(df[date_col], bins=bins, labels=labels, right=False)
    return df

def prepare_rater_factors_over_time_weekly(ratings_df, rater_df, rater_factor_col):
    # Add week column to ratings
    rater_df['week_dt'] = pd.to_datetime(rater_df['week_dt']).dt.normalize()
    ratings_df['week_dt'] = pd.to_datetime(ratings_df['createdAtMillis'], unit='ms').dt.to_period('W').dt.start_time

    
    # Ensure rater_df also have 'week' column
    if 'week_dt' not in rater_df.columns:
        raise ValueError("rater_df must include a 'week' column for weekly merging.")
    
    ratings_df = ratings_df.sort_values(['week_dt'])
    rater_df = rater_df.sort_values(["week_dt"])
    
    # Ensure participant IDs are the same type (string is usually safest)
    rater_df["raterParticipantId"] = rater_df["raterParticipantId"].astype(str)
    ratings_df["raterParticipantId"] = ratings_df["raterParticipantId"].astype(str)
    
    # Merge with rater factors by raterParticipantId and week
    df = pd.merge_asof(
        left=ratings_df,
        right=rater_df[["raterParticipantId", "week_dt", "raterFactor1", 'raterIntercept']],
        by="raterParticipantId",
        left_on="week_dt",
        right_on="week_dt",
        direction="backward",
        allow_exact_matches=False
    )

    # Convert labels
    df['helpful_binary'] = np.where(
        df['helpfulnessLevel'] == 'HELPFUL', 1,
        np.where(df['helpfulnessLevel'] == 'NOT_HELPFUL', 0, np.nan)
    )

    # # Clean up
    df = df.dropna(subset=['raterFactor1', 'helpful_binary'])

    # Label for coloring
    df['vote_label'] = df['helpful_binary'].map({1: 'Helpful', 0: 'Not Helpful'})

    return df

