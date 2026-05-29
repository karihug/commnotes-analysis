import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from src.data_loader import add_helpful_num
# from linearmodels.panel import PanelOLS

def add_prior_mean(group: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the running prior mean of noteFactor1 for a group of notes on the same tweet.
    """
    group = group.sort_values('note_order').copy()
    group['prior_mean'] = group['noteFactor1'].expanding().mean().shift(1)
    group['notealignment'] = group['noteFactor1'] * group['prior_mean']
    return group

def get_note_writing_preference(df: pd.DataFrame) -> str:
    """
    Determines a user's note-writing preference based on the partisanship of tweet authors.
    """
    dem_notes = df[df['party'] == 'democrat'].shape[0]
    rep_notes = df[df['party'] == 'republican'].shape[0]
    
    if dem_notes > rep_notes * 2:
        return "Primarily Writes Notes on Democrat Tweets"
    elif rep_notes > dem_notes * 2:
        return "Primarily Writes Notes on Republican Tweets"
    else:
        return "Mixed Note Writing"



def get_note_rating_preference(df: pd.DataFrame) -> str:
    """
    Determines a user's note-writing preference based on the partisanship of tweet authors.
    """
    dem_notes_pos = df[(df['party'] == 'democrat')&(df['helpfulnessLevel'].isin(['HELPFUL','SOMEWHAT_HELPFUL']))].shape[0]
    dem_notes_neg = df[(df['party'] == 'democrat')&(df['helpfulnessLevel'].isin(['NOT_HELPFUL']))].shape[0]


    rep_notes_pos = df[(df['party'] == 'republican')&(df['helpfulnessLevel'].isin(['HELPFUL','SOMEWHAT_HELPFUL']))].shape[0]
    rep_notes_neg = df[(df['party'] == 'republican')&(df['helpfulnessLevel'].isin(['NOT_HELPFUL']))].shape[0]
    
    party_agreement_score = ((dem_notes_pos + rep_notes_neg) - (rep_notes_pos + dem_notes_neg)) / (dem_notes_pos + rep_notes_neg + rep_notes_pos + dem_notes_neg)
    if party_agreement_score > .2:
        return "Primarily Agree with Democrat Tweets"
    elif party_agreement_score <-.2:
        return "Primarily Agree with Republican Tweets"
    else:
        return "Mixed Note Writing"
    
def run_its_analysis(weekly_correlation: pd.DataFrame, intervention_date_str: str = "2022-10-01") -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Performs an Interrupted Time Series (ITS) analysis.
    """
    its_df = weekly_correlation.copy()
    its_df['time'] = np.arange(len(its_df))
    intervention_date = pd.to_datetime(intervention_date_str)
    its_df['intervention'] = (its_df['week_start'] >= intervention_date).astype(int)
    its_df['time_after_intervention'] = its_df['time'] * its_df['intervention']

    X = its_df[['time', 'intervention', 'time_after_intervention']]
    X = sm.add_constant(X)
    y = its_df['correlation']
    
    return sm.OLS(y, X).fit()

def calculate_non_convergent_proportion(status_history_df: pd.DataFrame, notes_df: pd.DataFrame, time= "W") -> pd.DataFrame:
    """
    Calculates the weekly proportion of tweets that do not have a convergent note.
    """
    # Convert timestamp columns to datetime objects for easier filtering

    all_notes_and_tweets = pd.merge(status_history_df, notes_df, on='noteId', how='right', suffixes=('_status', '_note'))
    all_notes_and_tweets['week_dt'] = pd.to_datetime(all_notes_and_tweets['createdAtMillis_note'], unit='ms').dt.to_period('W').dt.start_time

    all_notes_and_tweets['note_creation_dt'] = pd.to_datetime(all_notes_and_tweets['createdAtMillis_note'], unit='ms')


    # Find the earliest note creation time for each unique tweetId once
    first_note_per_tweet = all_notes_and_tweets.groupby("tweetId")["note_creation_dt"].min()
    first_note_per_tweet.name = 'first_note_dt'

  
    # Identify all tweets that have at least one note with a final status
    helpful_notes = all_notes_and_tweets[
        all_notes_and_tweets["currentStatus"] == 'CURRENTLY_RATED_HELPFUL'
    ]
    unhelpful_notes = all_notes_and_tweets[
        all_notes_and_tweets["currentStatus"] == 'CURRENTLY_RATED_NOT_HELPFUL'
    ]

    # Find the earliest note creation time for these specific statuses
    first_helpful_note_time = helpful_notes.groupby('tweetId')['note_creation_dt'].min()
    first_unhelpful_note_time = unhelpful_notes.groupby('tweetId')['note_creation_dt'].min()


    # Resample the data to get weekly counts using the first note time as the index
    # This is a highly efficient way to get the denominator
    # The denominator is the number of tweets that got their FIRST note in the 14-day period.
    first_note_resampled = first_note_per_tweet.value_counts().resample(time).sum().fillna(0)

   # Resample the helpful and unhelpful data to get 14-day counts
    helpful_resampled = first_helpful_note_time.value_counts().resample(time).sum().fillna(0)
    unhelpful_resampled = first_unhelpful_note_time.value_counts().resample(time).sum().fillna(0)

    # We need to align the indexes of all series before calculating the proportion
    # This ensures that we are dividing counts for the correct time periods
    proportion_df = pd.DataFrame({
        'denominator_tweets': first_note_resampled,
        'helpful_tweets': helpful_resampled,
        'unhelpful_tweets': unhelpful_resampled
    }).fillna(0)

    # Calculate the proportions for helpful and unhelpful notes
    proportion_df['proportion_helpful'] = proportion_df['helpful_tweets'] / proportion_df['denominator_tweets']
    proportion_df['proportion_unhelpful'] = proportion_df['unhelpful_tweets'] / proportion_df['denominator_tweets']

    # Reset the index and name the new column 'week_dt'
    proportion_df = proportion_df.reset_index().rename(columns={'index': 'week_dt'})
    return proportion_df

def calculate_monthly_logit_coeffs(ratings_df: pd.DataFrame, note_df: pd.DataFrame, rater_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the logistic regression coefficient for the dot product predicting helpfulness, month by month.
    """
    df = pd.merge(ratings_df, note_df[['noteId', 'noteFactor1']], on='noteId')
    df = pd.merge(df, rater_df[['raterParticipantId', 'raterFactor1']], on='raterParticipantId')
    df = add_helpful_num(df)
    df['dot_product'] = df['raterFactor1'] * df['noteFactor1']
    df['month'] = pd.to_datetime(df['createdAtMillis'], unit='ms').dt.to_period('M').dt.start_time
    
    results = []
    for month, group in tqdm(df.groupby('month'), desc="Running monthly logistic regressions"):
        y = group['helpfulNum'].apply(lambda x: 1 if x > 0.5 else 0)
        X = group['dot_product']
        X = sm.add_constant(X)
        
        try:
            model = sm.Logit(y, X).fit(disp=0)
            results.append({'month': month, 'coefficient': model.params['dot_product'], 'p_value': model.pvalues['dot_product']})
        except Exception:
            results.append({'month': month, 'coefficient': np.nan, 'p_value': np.nan})
            
    return pd.DataFrame(results)



from typing import Callable, Dict
import numpy as np
import pandas as pd

# ---- Constants & tiny helpers ------------------------------------------------
PARTIES = ['democrat', 'republican']
CUTOFF_DATE = pd.Timestamp("2022-10-01")


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce')


def latest_by_key(df: pd.DataFrame, key: str, time_col: str, keep_cols: list[str]) -> pd.DataFrame:
    """For each `key`, keep the last by `time_col`, return only [key] + keep_cols."""
    out = df.copy()
    out[time_col] = _to_dt(out[time_col])
    cols = [key] + [c for c in keep_cols if c != key]
    return (
        out.sort_values(time_col)
           .drop_duplicates(key, keep='last')
           .loc[:, cols]
           .reset_index(drop=True)
    )


# ---- Base tables -------------------------------------------------------------
def build_merged_base(paper_df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    """Join paper + notes once; reuse everywhere."""
    return paper_df.merge(notes_df, left_on="note_id", right_on="noteId", how="inner")


def compute_latest_rater_factors(rater_df: pd.DataFrame) -> pd.DataFrame:
    return latest_by_key(
        rater_df, key='raterParticipantId', time_col='week_dt',
        keep_cols=['raterParticipantId', 'raterFactor1']
    )


def compute_latest_note_factors(note_df_full: pd.DataFrame) -> pd.DataFrame:
    return latest_by_key(
        note_df_full, key='noteId', time_col='week_dt',
        keep_cols=['noteId', 'noteFactor1']
    )


# ---- Part 1: Factor Similarity ----------------------------------------------
def compute_contributor_partisanship(
    merged_df: pd.DataFrame,
    latest_rater_factors: pd.DataFrame
) -> pd.DataFrame:
    """Return wide DF with author factor by party + author's own factor."""
    p1 = merged_df.merge(
        latest_rater_factors[['raterParticipantId', 'raterFactor1']],
        left_on='noteAuthorParticipantId', right_on='raterParticipantId', how='inner'
    )
    p1 = p1[p1['party'].isin(PARTIES)]
    contrib = (
        p1.groupby(['noteAuthorParticipantId', 'party'])['raterFactor1']
          .mean()
          .unstack()
          .reindex(columns=PARTIES)
    )
    contrib.columns = ['Democrat_Tweet_Author_Factor', 'Republican_Tweet_Author_Factor']
    contrib = contrib.merge(
        latest_rater_factors.rename(columns={'raterFactor1': 'Note_Author_Factor'}),
        left_index=True, right_on='raterParticipantId', how='inner'
    )
    return contrib


# ---- Part 2: Factor Migration ------------------------------------------------
def compute_rater_df_with_cohorts(
    rater_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    get_note_writing_preference: Callable[[pd.DataFrame], str],
    cutoff_date: pd.Timestamp = CUTOFF_DATE
) -> pd.DataFrame:
    """Attach author's writing cohort to raters and add Before/After period."""
    writing_cohorts = (
        merged_df.groupby('noteAuthorParticipantId')
                 .apply(get_note_writing_preference)
                 .rename('Writing_Cohort')
                 .reset_index()
    )
    out = rater_df.merge(
        writing_cohorts, left_on='raterParticipantId', right_on='noteAuthorParticipantId', how='inner'
    ).drop(columns=['noteAuthorParticipantId'])
    out['week_dt'] = _to_dt(out['week_dt'])
    out['Period'] = np.where(out['week_dt'] < cutoff_date, 'Before', 'After')
    return out


# ---- Part 3: Flagging Concentration -----------------------------------------
def compute_concentration_with_factors(
    merged_df: pd.DataFrame,
    latest_rater_factors: pd.DataFrame
) -> pd.DataFrame:
    df = merged_df[merged_df['party'].isin(PARTIES)]
    party_counts = (
        df.groupby(['noteAuthorParticipantId', 'party'])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=PARTIES, fill_value=0)
    )
    party_counts = party_counts.assign(
        total_notes=lambda d: d.sum(axis=1),
        concentration_score=lambda d: d[PARTIES].max(axis=1) / d['total_notes'],
        dominant_party=lambda d: np.where(d['democrat'] > d['republican'], 'Democrat', 'Republican')
    )
    return party_counts.merge(
        latest_rater_factors[['raterParticipantId', 'raterFactor1']],
        left_index=True, right_on='raterParticipantId', how='inner'
    )


# ---- Part 4: Interrupted Time Series ----------------------------------------
def compute_weekly_correlation(
    merged_df: pd.DataFrame,
    latest_rater_factors: pd.DataFrame,
    latest_note_factors: pd.DataFrame,
    notes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Weekly corr between raterFactor1 (author) and noteFactor1 (note),
    indexed by note creation time from notes_df['createdAtMillis'].
    """
    temporal = (
        merged_df
        .merge(latest_rater_factors[['raterParticipantId', 'raterFactor1']],
               left_on='noteAuthorParticipantId', right_on='raterParticipantId', how='inner')
        .merge(latest_note_factors[['noteId', 'noteFactor1']], on='noteId', how='inner')
        .merge(notes_df[['noteId', 'createdAtMillis']], on='noteId', how='left')
    )
    temporal['createdAt'] = pd.to_datetime(temporal['createdAtMillis'], unit='ms', errors='coerce')
    temporal = temporal.dropna(subset=['createdAt']).set_index('createdAt').sort_index()

    weekly = (
        temporal.resample('W')[['raterFactor1', 'noteFactor1']]
                .apply(lambda x: x['raterFactor1'].corr(x['noteFactor1']))
                .rename('correlation')
                .dropna()
                .reset_index()
                .rename(columns={'createdAt': 'week_start'})
    )
    return weekly


# A simple, drop‑in ITS (replace with your own if you already have one)
def run_its_analysis(weekly_correlation: pd.DataFrame, intervention: pd.Timestamp = CUTOFF_DATE):
    """
    OLS ITS with level + trend change and HAC errors.
    weekly_correlation: DataFrame with ['week_start', 'correlation'].
    """
    import statsmodels.api as sm

    df = weekly_correlation.copy()
    df['week_start'] = _to_dt(df['week_start'])
    df = df.sort_values('week_start').reset_index(drop=True)
    df['t'] = np.arange(len(df))
    df['post'] = (df['week_start'] >= intervention).astype(int)
    df['t_post'] = df['t'] * df['post']

    X = sm.add_constant(df[['t', 'post', 't_post']])
    y = df['correlation']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    return model


# ---- Part 5: Choice vs Algorithm --------------------------------------------
def compute_choice_vs_algorithm_df(
    ratings_df: pd.DataFrame,
    latest_note_factors: pd.DataFrame,
    merged_df: pd.DataFrame
) -> pd.DataFrame:
    ratings_df = ratings_df.copy()
    ratings_df['noteId'] = ratings_df['noteId'].astype(int, errors='ignore')

    rated = ratings_df.merge(latest_note_factors[['noteId', 'noteFactor1']], on='noteId', how='inner')
    avg_rated = (
        rated.groupby('raterParticipantId', as_index=False)['noteFactor1']
             .mean()
             .rename(columns={'noteFactor1': 'avg_rated_note_factor'})
    )

    written = merged_df.merge(latest_note_factors[['noteId', 'noteFactor1']], on='noteId', how='inner')
    avg_written = (
        written.groupby('noteAuthorParticipantId', as_index=False)['noteFactor1']
               .mean()
               .rename(columns={
                   'noteFactor1': 'avg_written_note_factor',
                   'noteAuthorParticipantId': 'raterParticipantId'
               })
    )
    return avg_rated.merge(avg_written, on='raterParticipantId', how='inner')


# ---- Convenience aggregator (optional) --------------------------------------
def compute_all(
    paper_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    rater_df: pd.DataFrame,
    note_df_full: pd.DataFrame,
    ratings_df: pd.DataFrame,
    get_note_writing_preference: Callable[[pd.DataFrame], str],
) -> Dict[str, pd.DataFrame]:
    merged = build_merged_base(paper_df, notes_df)
    latest_raters = compute_latest_rater_factors(rater_df)
    latest_notes = compute_latest_note_factors(note_df_full)

    return {
        "contributor_partisanship": compute_contributor_partisanship(merged, latest_raters),
        "rater_df_with_cohorts": compute_rater_df_with_cohorts(rater_df, merged, get_note_writing_preference),
        "concentration_with_factors": compute_concentration_with_factors(merged, latest_raters),
        "weekly_correlation": compute_weekly_correlation(merged, latest_raters, latest_notes, notes_df),
        "choice_vs_algorithm": compute_choice_vs_algorithm_df(ratings_df, latest_notes, merged),
    }

