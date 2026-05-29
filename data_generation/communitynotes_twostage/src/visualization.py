import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import scipy.stats as st


def plot_factor_similarity(df: pd.DataFrame):
    """
    Plots the relationship between a contributor's factor and the average factor of authors they write notes on.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Note_Author_Factor', y='Democrat_Tweet_Author_Factor', alpha=0.5, label='Wrote Note on Democrat Tweet')
    sns.scatterplot(data=df, x='Note_Author_Factor', y='Republican_Tweet_Author_Factor', alpha=0.5, label='Wrote Note on Republican Tweet')
    plt.title("Note Author's Factor vs. Average Factor of Tweet Authors They Write Notes On")
    plt.xlabel("Note Author's Own raterFactor1")
    plt.ylabel("Average raterFactor1 of Tweet Authors")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_factor_migration(df: pd.DataFrame):
    """
    Creates a violin plot to show the migration of rater factors over time.
    """
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, x='Writing_Cohort', y='raterFactor1', hue='Period', split=True)
    plt.title('Factor Migration by Note-Writing Cohort (Before vs. After Oct 2022)')
    plt.xlabel('User Note-Writing Behavior')
    plt.ylabel('raterFactor1 Distribution')
    plt.grid(True)
    plt.show()

def plot_concentration_analysis(df: pd.DataFrame):
    """
    Creates plots to analyze flagging concentration.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='concentration_score', bins=20, kde=True)
    plt.title('Distribution of User Flagging Concentration')
    plt.xlabel('Concentration Score (0.5 = Mixed, 1.0 = Partisan)')
    plt.ylabel('Number of Users')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, x='raterFactor1', y='concentration_score', hue='dominant_party',
        palette={'Democrat': 'blue', 'Republican': 'red'}, alpha=0.6
    )
    plt.title('User Factor vs. Flagging Concentration')
    plt.xlabel('User Factor (raterFactor1)')
    plt.ylabel('Flagging Concentration Score')
    plt.axhline(0.5, color='grey', linestyle='--', label='Perfectly Mixed')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_temporal_correlation(weekly_correlation_df: pd.DataFrame):
    """
    Plots a variable over time, including the intervention line.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(weekly_correlation_df['week_start'], weekly_correlation_df['correlation'], marker='o', linestyle='-', label='Weekly Correlation')
    plt.axvline(pd.to_datetime("2022-10-01"), color='red', linestyle='--', label='Algorithm Change (Oct 2022)')
    plt.title('Correlation Between Note Author Factor and Note Factor Over Time')
    plt.xlabel('Week')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_mean_by_quantile(df: pd.DataFrame, plot_var: str):
    """
    Plots the mean of a variable over time, grouped by note count quantile categories.
    """
    df1 = df.copy()
    if 'note_count' not in df1.columns:
        print(f"Error: 'note_count' column not found for plotting '{plot_var}'.")
        return
        
    df1['quantile'] = pd.qcut(df1['note_count'], 4, labels=False, duplicates='drop')
    
    HEAVY, MEDIUM, LIGHT = [3], [2], [0, 1]
    plt.figure(figsize=(15, 7))

    for label, quantiles, color in [('HEAVY', HEAVY, 'red'), ('MEDIUM', MEDIUM, 'orange'), ('LIGHT', LIGHT, 'green')]:
        mask = df1['quantile'].isin(quantiles)
        data = df1[mask].groupby('week_dt')[plot_var].mean().sort_index()
        plt.scatter(data.index, data.values, label=label, color=color, alpha=0.7)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xlabel('Week')
    plt.ylabel(f'Mean {plot_var}')
    plt.title(f'Mean {plot_var} Over Time by Note Count Quantile')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_metric_over_time(df: pd.DataFrame, y_var: str, title: str):
    """
    Generic function to plot a metric over time.
    """
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=df, x='week_dt', y=y_var, marker='o')
    plt.title(title)
    plt.xlabel('Week')
    plt.ylabel(y_var)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from rdrobust import rdrobust, rdplot

def plot_factor_distribution_shifts(df: pd.DataFrame, factor_col: str):
    """
    Plots the density of a factor column, grouped by phase.
    """
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=df, x=factor_col, hue='phase', common_norm=False, linewidth=2)
    plt.title(f'Shift in {factor_col} Distribution Over Time')
    plt.axvline(0, color='grey', linestyle='--')
    plt.grid(True)
    plt.show()

def plot_partisan_trends(df: pd.DataFrame, title: str):
    """
    Plots the fraction of notes over time for positive vs. negative factor users.
    """
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df[df['factor_group'] == 'Positive Factor'], x='month', y='fraction', label='Positive Factor (Left)', color='blue')
    sns.lineplot(data=df[df['factor_group'] == 'Negative Factor'], x='month', y='fraction', label='Negative Factor (Right)', color='red')
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Fraction of Notes')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

def plot_intercept_vs_factor(df: pd.DataFrame):
    """
    Creates a scatter plot of noteIntercept vs. noteFactor.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='noteFactor1', y='noteIntercept', alpha=0.1)
    plt.title('Note Intercept vs. Note Factor')
    plt.xlabel('Note Factor')
    plt.ylabel('Note Intercept')
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd

# --- 2023 Community Notes milestones to mark on the charts ---
EVENTS_2023 = [
    # ("2023-01-15", "Stability update"),
    # ("2023-02-20", "Coverage model"),
    # ("2023-04-20", "Confidence score (+Helpful)"),
    # # ("2023-05-31", "Notes on images"),
    # # ("2023-08-29", "Reduce flipping"),
    # # ("2023-09-06", "Notes on videos"),
    # ("2023-10-12", "Speed/distribution"),
    # ("2023-10-29", "Demonetization"),
    # ("2023-11-15", "Global expansion"),
]

def _add_event_markers(ax, events=None, rotation=90, pad_frac=0.08,
                       line_kwargs=None, text_kwargs=None):
    """
    Add vertical lines + labels for key dates.
    pad_frac adds headroom so labels don't collide with the top of the plot.
    """
    if events is None:
        events = EVENTS_2023
    if line_kwargs is None:
        line_kwargs = dict(linestyle='--', linewidth=1, alpha=0.6)
    if text_kwargs is None:
        text_kwargs = dict(rotation=rotation, ha='left', va='bottom', fontsize=8)

    ymin, ymax = ax.get_ylim()
    if ymax <= 0:
        ymax = 1.0
    # add a bit of headroom for labels
    ax.set_ylim(ymin, ymax * (1 + pad_frac))
    y_for_text = ax.get_ylim()[1] * (1 - pad_frac * 0.2)

    for date_str, label in events:
        dt = pd.to_datetime(date_str)
        ax.axvline(dt, **line_kwargs)
        ax.annotate(label, xy=(dt, y_for_text), xytext=(2, 2),
                    textcoords='offset points', **text_kwargs)

def plot_non_convergent_trends(df: pd.DataFrame, events=None):
    """
    Plots the proportion of tweets with helpful and unhelpful notes over time,
    with optional milestone markers. Also plots 'no convergent note'.
    """
    d = df.copy()
    d['week_dt'] = pd.to_datetime(d['week_dt'])

    # --- Chart 1: Helpful vs Unhelpful ---
    plt.figure(figsize=(15, 7))
    plt.plot(d['week_dt'], d['proportion_helpful'], marker='o', label='Helpful Notes')
    plt.plot(d['week_dt'], d['proportion_unhelpful'], marker='x', label='Unhelpful Notes')
    ax = plt.gca()
    _add_event_markers(ax, events)  # <-- add markers here
    plt.title('Proportion of Tweets with Final Status Over Time (14-day window)')
    plt.xlabel('Date')
    plt.ylabel('Proportion')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Chart 2: No convergent note ---
    d['proportion_no_convergent_note'] = 1 - d['proportion_helpful'] - d['proportion_unhelpful']
    d['rolling_avg'] = d['proportion_no_convergent_note'].rolling(window=3, center=True).mean()

    plt.figure(figsize=(15, 7))
    plt.plot(d['week_dt'], d['proportion_no_convergent_note'], marker='o', alpha=0.5, label='Raw')
    plt.plot(d['week_dt'], d['rolling_avg'], label='3-Week Rolling Avg')
    ax2 = plt.gca()
    _add_event_markers(ax2, events)  # <-- add markers here
    plt.title('Proportion of Tweets with No Convergent Note')
    plt.xlabel('Month')
    plt.ylabel('Proportion')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_voting_dynamics_density(df: pd.DataFrame, title: str):
    """
    Creates a 2D density plot of raterFactor vs. noteFactor, colored by vote.
    """
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df, x='raterFactor1', y='noteFactor1', hue='vote_label', fill=True, alpha=0.5, common_norm=False)
    plt.title(title)
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.grid(True)
    plt.show()

def plot_logit_rdd(df: pd.DataFrame, intervention_date_str: str = "2022-10-01"):
    """
    Plots the result of the logistic coefficient RDD.
    """
    df['month_num'] = (df['month'] - df['month'].min()).dt.days // 30
    cutoff = (pd.to_datetime(intervention_date_str) - df['month'].min()).days // 30
    display(df)
    
    plt.figure(figsize=(14, 7))
    rdplot(y=df['coefficient'], x=df['month_num'], c=cutoff, title='Regression Discontinuity of Logit Coefficient',
           x_label='Month Number', y_label='Logit Coefficient (Dot Product -> Helpfulness)')
    plt.show()

def plot_alignment_vs_helpfulness(df: pd.DataFrame):
    """
    Plots note alignment vs. the proportion of helpful ratings.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='notealignment', y='proportion_helpful', alpha=0.3)
    plt.title('Proportion of Helpful Ratings vs. Note Alignment with Previous Notes')
    plt.xlabel('Note Alignment')
    plt.ylabel('Proportion')
    plt.grid(True)
    plt.show()

# (Include other plotting functions from the previous refactoring as well)
# plot_factor_similarity, plot_factor_migration, plot_concentration_analysis, 
# plot_temporal_correlation, plot_mean_by_quantile, plot_metric_over_time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_kde_ci_by_cohort(
    df,
    value_col="raterFactor1",
    cohort_col="Cohort",
    period_col="Period",
    cohorts=("Legacy Users", "New Users"),
    n_boot=200,
    ci=95,
    bw="scott",
    grid_percentiles=(0.5, 99.5),
    grid_points=400,
    filename="kde_ci_panels.png",
    random_state=None,
    verbose=True,
):
    """
    Two-panel KDE with bootstrap CIs by cohort using SciPy's gaussian_kde.
    Also prints group means for each cohort/period.
    NOTE: gaussian_kde is imported locally to avoid NameError in notebooks.
    """
    # ---- local import (guarantees availability in this scope) ----
    from scipy.stats import gaussian_kde as _gaussian_kde

    # --- Prepare data ---
    x_all = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy()
    if x_all.size < 2:
        raise ValueError("Not enough data to plot KDEs.")

    lo_p, hi_p = grid_percentiles
    lo, hi = np.nanpercentile(x_all, [lo_p, hi_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        eps = 1e-6 if np.isfinite(lo) else 1.0
        lo, hi = (float(np.nanmin(x_all)) - eps, float(np.nanmax(x_all)) + eps)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    grid = np.linspace(lo, hi, grid_points)

    # Periods & colors
    periods = sorted(pd.Series(df[period_col]).dropna().unique().tolist())
    cmap = plt.get_cmap("tab10")
    color = {p: cmap(i % 10) for i, p in enumerate(periods)}

    rng = np.random.default_rng(random_state)

    def eval_kde(x, grid):
        kde = _gaussian_kde(x, bw_method=bw)
        return kde(grid)

    # --- Figure layout ---
    fig, axes = plt.subplots(
        1, len(cohorts), figsize=(6 * len(cohorts), 4.5),
        sharex=True, sharey=True
    )
    if len(cohorts) == 1:
        axes = [axes]

    # --- Main loop ---
    for ax, cohort in zip(axes, cohorts):
        sub = df.loc[df[cohort_col] == cohort]
        for p in periods:
            x = pd.to_numeric(
                sub.loc[sub[period_col] == p, value_col],
                errors="coerce"
            ).dropna().to_numpy()
            if x.size < 2:
                continue

            # Bootstrap KDE for CI
            boots = np.empty((n_boot, grid.size), dtype=float)
            for b in range(n_boot):
                sample = x[rng.integers(0, x.size, size=x.size)]
                boots[b] = eval_kde(sample, grid)

            mean_curve = boots.mean(axis=0)
            alpha = (100 - ci) / 2.0
            lo_ci = np.percentile(boots, alpha, axis=0)
            hi_ci = np.percentile(boots, 100 - alpha, axis=0)

            ax.plot(grid, mean_curve, lw=2.0, color=color[p], label=str(p))
            ax.fill_between(grid, lo_ci, hi_ci, color=color[p], alpha=0.18, linewidth=0)

            # Period mean line (raw data mean)
            overall_mean = x.mean()
            ax.axvline(overall_mean, color=color[p], lw=1.6, linestyle="--", alpha=0.9)

            # Print means
            if verbose:
                print(f"[{cohort} | {p}]")
                print(f"  Raw data mean = {overall_mean:.3f}")
                print(f"  Mean of KDE curve = {mean_curve.mean():.6f}\n")

        ax.set_title(cohort, fontweight="bold")
        ax.set_xlabel(value_col)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Density")

    # Legend
    handles = [Line2D([0], [0], color=color[p], lw=2.0, label=p) for p in periods]
    mean_handle = Line2D([0], [0], color="black", lw=1.6, linestyle="--", label="mean")
    fig.legend(
        handles + [mean_handle],
        [h.get_label() for h in handles] + ["mean"],
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


