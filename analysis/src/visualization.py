import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import seaborn as sns
import scipy.stats as st
from matplotlib.lines import Line2D
from pathlib import Path

# ---------------------------------------------------------------------------
# Publication-style shared constants and helpers
# ---------------------------------------------------------------------------

# Shared palette for publication figures.
_C_BLUE   = "#1F77B4"   # pre-rollout / baseline / current method
_C_ORANGE = "#D85A30"   # post-rollout / proposed method
_C_EVENT  = "#D62728"   # intervention line
_C_ZERO   = "#666666"   # neutral reference lines
_C_LIGHT  = "#BBBBBB"
_C_PURPLE = "#534AB7"
_C_AMBER  = "#F4B445"

# Additional muted palette for multi-group plots (3+ groups)
_MUTED_PALETTE = [_C_BLUE, _C_ORANGE, "#6AAA96", _C_AMBER, _C_PURPLE]

_FONT_TITLE  = 11
_FONT_LABEL  = 10
_FONT_TICK   = 9
_FONT_LEGEND = 9
_FONT_ANNOT  = 8


def apply_publication_style():
    """Apply the shared figure style used by the sample.py templates."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.titlesize": _FONT_TITLE,
        "axes.titleweight": "normal",
        "axes.labelsize": _FONT_LABEL,
        "xtick.labelsize": _FONT_TICK,
        "ytick.labelsize": _FONT_TICK,
        "legend.fontsize": _FONT_LEGEND,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


apply_publication_style()


def _setup_ax(ax):
    """Apply shared publication axis aesthetics from the sample.py templates."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=_FONT_TICK, length=3, width=0.8)
    ax.grid(True, linewidth=0.5, alpha=0.25, color=_C_LIGHT)
    ax.set_axisbelow(True)


def _finalize(fig, ax_or_axes, xlabel=None, ylabel=None, title=None,
              save_path=None):
    """Set labels, optional suptitle, tight layout, and optional save."""
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in axes:
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=_FONT_LABEL)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=_FONT_LABEL)
    if title is not None:
        if len(axes) == 1:
            axes[0].set_title(title, loc="left", fontsize=_FONT_TITLE, pad=10)
        else:
            fig.suptitle(title, fontsize=_FONT_TITLE + 1, y=1.02)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_factor_similarity(df: pd.DataFrame):
    """
    Plots the relationship between a contributor's factor and the average factor
    of authors they write notes on.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(df['Note_Author_Factor'], df['Democrat_Tweet_Author_Factor'],
               color=_C_BLUE, alpha=0.45, s=20, linewidths=0,
               label='Wrote note on Democrat tweet', zorder=3)
    ax.scatter(df['Note_Author_Factor'], df['Republican_Tweet_Author_Factor'],
               color=_C_ORANGE, alpha=0.45, s=20, linewidths=0,
               label='Wrote note on Republican tweet', zorder=3)
    ax.legend(frameon=False, fontsize=_FONT_LEGEND, loc="upper right")
    _setup_ax(ax)
    _finalize(fig, ax,
              xlabel="Note author's raterFactor1",
              ylabel="Avg. raterFactor1 of tweet authors",
              title="Note author factor vs. avg. factor of tweet authors")


def plot_factor_migration(df: pd.DataFrame):
    """
    Violin plot showing the migration of rater factors over time.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.violinplot(data=df, x='Writing_Cohort', y='raterFactor1', hue='Period',
                   split=True, palette=[_C_BLUE, _C_ORANGE],
                   linewidth=0.8, inner="quartile", ax=ax)
    ax.legend(frameon=False, fontsize=_FONT_LEGEND, loc="upper right")
    _setup_ax(ax)
    _finalize(fig, ax,
              xlabel="User note-writing behavior",
              ylabel="raterFactor1 distribution",
              title="Factor migration by note-writing cohort (before vs. after Oct 2022)")


def plot_concentration_analysis(df: pd.DataFrame):
    """
    Histogram of flagging concentration and scatter of factor vs. concentration.
    """
    # Panel 1 – distribution
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.histplot(data=df, x='concentration_score', bins=20, kde=True,
                 color=_C_BLUE, alpha=0.6, line_kws={"lw": 1.8}, ax=ax)
    _setup_ax(ax)
    _finalize(fig, ax,
              xlabel="Concentration score (0.5 = mixed, 1.0 = partisan)",
              ylabel="Number of users",
              title="Distribution of user flagging concentration")

    # Panel 2 – scatter
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
    pal = {'Democrat': _C_BLUE, 'Republican': _C_ORANGE}
    sns.scatterplot(data=df, x='raterFactor1', y='concentration_score',
                    hue='dominant_party', palette=pal,
                    alpha=0.5, s=20, linewidths=0, ax=ax2)
    ax2.axhline(0.5, color=_C_ZERO, linewidth=1, linestyle='--', label='Perfectly mixed')
    ax2.legend(frameon=False, fontsize=_FONT_LEGEND, loc="upper right")
    _setup_ax(ax2)
    _finalize(fig2, ax2,
              xlabel="User factor (raterFactor1)",
              ylabel="Flagging concentration score",
              title="User factor vs. flagging concentration")


def plot_temporal_correlation(weekly_correlation_df: pd.DataFrame):
    """
    Plots a correlation metric over time with an intervention marker.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(weekly_correlation_df['week_start'],
               weekly_correlation_df['correlation'],
               color=_C_BLUE, alpha=0.55, s=22, linewidths=0, zorder=3)
    ax.plot(weekly_correlation_df['week_start'],
            weekly_correlation_df['correlation'],
            color=_C_BLUE, linewidth=1.6, alpha=0.7)
    ax.axvline(pd.to_datetime("2022-10-01"), color=_C_EVENT,
               linewidth=1.2, linestyle='--', label='Algorithm change (Oct 2022)')
    ax.legend(frameon=False, fontsize=_FONT_LEGEND, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    _setup_ax(ax)
    _finalize(fig, ax,
              xlabel="Week",
              ylabel="Pearson correlation",
              title="Correlation between note author factor and note factor over time")


def plot_mean_by_quantile(df: pd.DataFrame, plot_var: str):
    """
    Plots the mean of a variable over time, grouped by note count quantile.
    """
    df1 = df.copy()
    if 'note_count' not in df1.columns:
        print(f"Error: 'note_count' column not found for plotting '{plot_var}'.")
        return

    df1['quantile'] = pd.qcut(df1['note_count'], 4, labels=False, duplicates='drop')

    HEAVY, MEDIUM, LIGHT = [3], [2], [0, 1]
    groups = [('Heavy', HEAVY, _MUTED_PALETTE[0]),
              ('Medium', MEDIUM, _MUTED_PALETTE[2]),
              ('Light', LIGHT, _MUTED_PALETTE[3])]

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, quantiles, color in groups:
        mask = df1['quantile'].isin(quantiles)
        data = df1[mask].groupby('week_dt')[plot_var].mean().sort_index()
        ax.scatter(data.index, data.values, label=label, color=color,
                   alpha=0.6, s=22, linewidths=0, zorder=3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    ax.legend(frameon=False, fontsize=_FONT_LEGEND, loc="upper right")
    _setup_ax(ax)
    _finalize(fig, ax,
              xlabel="Week",
              ylabel=f"Mean {plot_var}",
              title=f"Mean {plot_var} over time by note count quantile")


def plot_metric_over_time(df: pd.DataFrame, y_var: str, title: str):
    """
    Generic function to plot a metric over time.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df['week_dt'], df[y_var],
               color=_C_BLUE, alpha=0.55, s=22, linewidths=0, zorder=3)
    ax.plot(df['week_dt'], df[y_var],
            color=_C_BLUE, linewidth=1.6, alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    _setup_ax(ax)
    _finalize(fig, ax, xlabel="Week", ylabel=y_var, title=title)


from rdrobust import rdrobust, rdplot


def plot_factor_distribution_shifts(df: pd.DataFrame, factor_col: str,
                                    save_path: str = None, title: str = None):
    """
    Plots the density of a factor column, grouped by phase.
    """
    _FACTOR_LABELS = {"raterFactor1": "Rater Factor", "noteFactor1": "Note Factor"}
    xlabel = _FACTOR_LABELS.get(factor_col, factor_col)

    fig, ax = plt.subplots(figsize=(6.25, 4.25))

    phases = sorted(df["phase"].dropna().unique())
    colors = [_C_BLUE, _C_ORANGE] + _MUTED_PALETTE[2:]

    values_by_phase = [
        pd.to_numeric(
            df.loc[df["phase"] == phase, factor_col], errors="coerce"
        ).dropna().to_numpy()
        for phase in phases
    ]

    if len(phases) == 2 and all(values.size > 1 for values in values_by_phase):
        x_all = np.concatenate(values_by_phase)
        x_min, x_max = np.nanpercentile(x_all, [0.5, 99.5])
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
        pad = (x_max - x_min) * 0.08
        grid = np.linspace(x_min - pad, x_max + pad, 400)

        curves = [
            st.gaussian_kde(values, bw_method=0.18)(grid)
            for values in values_by_phase
        ]
        above = curves[1] >= curves[0]
        ax.fill_between(
            grid, curves[0], curves[1],
            where=above, color=colors[1], alpha=0.18, linewidth=0
        )
        ax.fill_between(
            grid, curves[1], curves[0],
            where=~above, color=colors[0], alpha=0.18, linewidth=0
        )
        for phase, curve, color in zip(phases, curves, colors):
            ax.plot(grid, curve, color=color, linewidth=2.2, label=str(phase))
    else:
        for phase, color in zip(phases, colors):
            sub = pd.to_numeric(
                df.loc[df["phase"] == phase, factor_col], errors="coerce"
            ).dropna()
            sns.kdeplot(
                sub,
                ax=ax,
                color=color,
                linewidth=2.2,
                label=str(phase),
                common_norm=False,
            )

    ax.axvline(0, color=_C_ZERO, linewidth=0.6, linestyle=":", zorder=1)

    ax.legend(
        frameon=False,
        fontsize=_FONT_LEGEND,
        loc="upper right",
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.3,
    )

    _setup_ax(ax)
    _finalize(
        fig,
        ax,
        xlabel=xlabel,
        ylabel="Density",
        title=title if title else f"Shift in {xlabel} Distribution Over Time",
        save_path=save_path,
    )


def plot_partisan_trends(df: pd.DataFrame, title: str):
    """
    Plots the fraction of notes over time for positive vs. negative factor users.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for group, color, label in [
        ('Positive Factor', _C_BLUE,   'Positive factor (left)'),
        ('Negative Factor', _C_ORANGE, 'Negative factor (right)'),
    ]:
        sub = df[df['factor_group'] == group]
        ax.plot(sub['month'], sub['fraction'], color=color,
                linewidth=2, label=label)
        ax.scatter(sub['month'], sub['fraction'], color=color,
                   alpha=0.55, s=22, linewidths=0, zorder=3)

    ax.legend(frameon=False, fontsize=_FONT_LEGEND, loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    _setup_ax(ax)
    _finalize(fig, ax, xlabel="Month", ylabel="Fraction of notes", title=title)


def plot_intercept_vs_factor(df: pd.DataFrame):
    """
    Scatter plot of noteIntercept vs. noteFactor.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(df['noteFactor1'], df['noteIntercept'],
               color=_C_BLUE, alpha=0.12, s=16, linewidths=0)
    _setup_ax(ax)
    _finalize(fig, ax,
              xlabel="Note factor",
              ylabel="Note intercept",
              title="Note intercept vs. note factor")


# --- 2023 Community Notes milestones to mark on the charts ---
EVENTS_2023 = [
    # ("2023-01-15", "Stability update"),
    # ("2023-02-20", "Coverage model"),
    # ("2023-04-20", "Confidence score (+Helpful)"),
    # ("2023-10-12", "Speed/distribution"),
    # ("2023-10-29", "Demonetization"),
    # ("2023-11-15", "Global expansion"),
]


def _add_event_markers(ax, events=None, rotation=90, pad_frac=0.08,
                       line_kwargs=None, text_kwargs=None):
    """Add vertical lines + labels for key dates."""
    if events is None:
        events = EVENTS_2023
    if line_kwargs is None:
        line_kwargs = dict(color=_C_ZERO, linestyle='--', linewidth=0.9, alpha=0.7)
    if text_kwargs is None:
        text_kwargs = dict(rotation=rotation, ha='left', va='bottom',
                           fontsize=_FONT_ANNOT)

    ymin, ymax = ax.get_ylim()
    if ymax <= 0:
        ymax = 1.0
    ax.set_ylim(ymin, ymax * (1 + pad_frac))
    y_for_text = ax.get_ylim()[1] * (1 - pad_frac * 0.2)

    for date_str, label in events:
        dt = pd.to_datetime(date_str)
        ax.axvline(dt, **line_kwargs)
        ax.annotate(label, xy=(dt, y_for_text), xytext=(2, 2),
                    textcoords='offset points', **text_kwargs)


def plot_non_convergent_trends(df: pd.DataFrame, events=None):
    """
    Proportion of tweets with helpful/unhelpful notes and no convergent note,
    with optional milestone markers.
    """
    d = df.copy()
    d['week_dt'] = pd.to_datetime(d['week_dt'])

    # Chart 1: Helpful vs Unhelpful
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d['week_dt'], d['proportion_helpful'],
            color=_C_BLUE, linewidth=2, label='Helpful notes')
    ax.scatter(d['week_dt'], d['proportion_helpful'],
               color=_C_BLUE, alpha=0.55, s=22, linewidths=0, zorder=3)
    ax.plot(d['week_dt'], d['proportion_unhelpful'],
            color=_C_ORANGE, linewidth=2, label='Unhelpful notes')
    ax.scatter(d['week_dt'], d['proportion_unhelpful'],
               color=_C_ORANGE, alpha=0.55, s=22, linewidths=0, zorder=3)
    _add_event_markers(ax, events)
    ax.legend(frameon=False, fontsize=_FONT_LEGEND, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    _setup_ax(ax)
    _finalize(fig, ax,
              xlabel="Date",
              ylabel="Proportion",
              title="Proportion of tweets with final status over time (14-day window)")

    # Chart 2: No convergent note
    d['proportion_no_convergent_note'] = (
        1 - d['proportion_helpful'] - d['proportion_unhelpful']
    )
    d['rolling_avg'] = d['proportion_no_convergent_note'].rolling(
        window=3, center=True).mean()

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(d['week_dt'], d['proportion_no_convergent_note'],
                color=_C_BLUE, alpha=0.45, s=22, linewidths=0, zorder=3, label='Raw')
    ax2.plot(d['week_dt'], d['rolling_avg'],
             color=_C_BLUE, linewidth=2, label='3-week rolling avg')
    _add_event_markers(ax2, events)
    ax2.legend(frameon=False, fontsize=_FONT_LEGEND, loc="upper right")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.get_xticklabels(), rotation=35, ha='right')
    _setup_ax(ax2)
    _finalize(fig2, ax2,
              xlabel="Month",
              ylabel="Proportion",
              title="Proportion of tweets with no convergent note")


def plot_voting_dynamics_density(df: pd.DataFrame, title: str):
    """
    2D KDE of raterFactor vs. noteFactor, colored by vote.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.kdeplot(data=df, x='raterFactor1', y='noteFactor1',
                hue='vote_label', fill=True, alpha=0.35,
                common_norm=False, linewidths=1.2, ax=ax)
    ax.axhline(0, color=_C_ZERO, linewidth=0.9, linestyle='--')
    ax.axvline(0, color=_C_ZERO, linewidth=0.9, linestyle='--')
    ax.legend(frameon=False, fontsize=_FONT_LEGEND, loc="upper right")
    _setup_ax(ax)
    _finalize(fig, ax,
              xlabel="Rater factor",
              ylabel="Note factor",
              title=title)


def plot_logit_rdd(df: pd.DataFrame, intervention_date_str: str = "2022-10-01"):
    """
    Plots the result of the logistic coefficient RDD.
    """
    df['month_num'] = (df['month'] - df['month'].min()).dt.days // 30
    cutoff = (pd.to_datetime(intervention_date_str) - df['month'].min()).days // 30
    display(df)

    plt.figure(figsize=(7, 4))
    rdplot(y=df['coefficient'], x=df['month_num'], c=cutoff,
           title='Regression discontinuity of logit coefficient',
           x_label='Month number',
           y_label='Logit coefficient (dot product → helpfulness)')
    # apply publication spine cleanup to the active axes after rdplot draws
    _setup_ax(plt.gca())
    plt.tight_layout()
    plt.show()


def plot_alignment_vs_helpfulness(df: pd.DataFrame):
    """
    Scatter of note alignment vs. proportion of helpful ratings.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(df['notealignment'], df['proportion_helpful'],
               color=_C_BLUE, alpha=0.3, s=18, linewidths=0)
    _setup_ax(ax)
    _finalize(fig, ax,
              xlabel="Note alignment",
              ylabel="Proportion helpful",
              title="Proportion of helpful ratings vs. note alignment")


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
    """
    from scipy.stats import gaussian_kde as _gaussian_kde

    x_all = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy()
    if x_all.size < 2:
        raise ValueError("Not enough data to plot KDEs.")

    lo_p, hi_p = grid_percentiles
    lo, hi = np.nanpercentile(x_all, [lo_p, hi_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        eps = 1e-6 if np.isfinite(lo) else 1.0
        lo, hi = float(np.nanmin(x_all)) - eps, float(np.nanmax(x_all)) + eps
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    grid = np.linspace(lo, hi, grid_points)

    periods = sorted(pd.Series(df[period_col]).dropna().unique().tolist())
    period_colors = [_C_BLUE, _C_ORANGE] + _MUTED_PALETTE[2:]
    color = {p: period_colors[i % len(period_colors)] for i, p in enumerate(periods)}

    rng = np.random.default_rng(random_state)

    def eval_kde(x, g):
        return _gaussian_kde(x, bw_method=bw)(g)

    fig, axes = plt.subplots(
        1, len(cohorts),
        figsize=(5.5 * len(cohorts), 4),
        sharex=True, sharey=True
    )
    if len(cohorts) == 1:
        axes = [axes]

    for ax, cohort in zip(axes, cohorts):
        sub = df.loc[df[cohort_col] == cohort]
        for p in periods:
            x = pd.to_numeric(
                sub.loc[sub[period_col] == p, value_col], errors="coerce"
            ).dropna().to_numpy()
            if x.size < 2:
                continue

            boots = np.empty((n_boot, grid.size), dtype=float)
            for b in range(n_boot):
                sample = x[rng.integers(0, x.size, size=x.size)]
                boots[b] = eval_kde(sample, grid)

            mean_curve = boots.mean(axis=0)
            alpha_ci = (100 - ci) / 2.0
            lo_ci = np.percentile(boots, alpha_ci, axis=0)
            hi_ci = np.percentile(boots, 100 - alpha_ci, axis=0)

            ax.plot(grid, mean_curve, lw=2.0, color=color[p], label=str(p))
            ax.fill_between(grid, lo_ci, hi_ci, color=color[p],
                            alpha=0.18, linewidth=0)

            overall_mean = x.mean()
            ax.axvline(overall_mean, color=color[p], lw=1.4,
                       linestyle="--", alpha=0.9)

            if verbose:
                print(f"[{cohort} | {p}]")
                print(f"  Raw data mean = {overall_mean:.3f}")
                print(f"  Mean of KDE curve = {mean_curve.mean():.6f}\n")

        ax.set_title(cohort, fontsize=_FONT_TITLE, fontweight="normal")
        ax.set_xlabel(value_col, fontsize=_FONT_LABEL)
        _setup_ax(ax)

    axes[0].set_ylabel("Density", fontsize=_FONT_LABEL)

    handles = [Line2D([0], [0], color=color[p], lw=2.0, label=p) for p in periods]
    mean_handle = Line2D([0], [0], color="black", lw=1.4, linestyle="--", label="mean")
    fig.legend(
        handles + [mean_handle],
        [h.get_label() for h in handles] + ["mean"],
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        frameon=False,
        fontsize=_FONT_LEGEND,
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])
