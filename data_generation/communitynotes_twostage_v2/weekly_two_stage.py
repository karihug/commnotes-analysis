#!/usr/bin/env python3
"""
This writes one metrics CSV for plotting and saves the final fitted note/rater
parameters for both the baseline X model and the low-rank-variance weighted X
model.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TWOSTAGE_DIR = SCRIPT_DIR.parent / "communitynotes_twostage"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TWOSTAGE_DIR) not in sys.path:
    sys.path.insert(0, str(TWOSTAGE_DIR))

import constants as c
import helper as mc
import utility
from helper import predict_ratings_with_factors, run_weighted_x_matrix_factorization


DEFAULT_RATINGS_PATH = (
    REPO_ROOT / "two-stage-experiments/data_input/ratings_2023_01_01_to_2024_06_01.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data_output" / "good_two_stage_runs"


def _suppress_stdout_if_needed(verbose: bool):
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def _default_run_label() -> str:
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        return f"job_{os.environ['SLURM_ARRAY_JOB_ID']}_task_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    if os.environ.get("SLURM_JOB_ID"):
        return f"job_{os.environ['SLURM_JOB_ID']}"
    return pd.Timestamp.now().strftime("manual_%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _summarize_predictions(
    ratings: pd.DataFrame,
    *,
    model: str,
    cutoff: pd.Timestamp,
    prediction_end: pd.Timestamp,
    global_bias: float,
    note_params: pd.DataFrame,
    rater_params: pd.DataFrame,
    extra: dict[str, Any],
) -> dict[str, Any]:
    preds = predict_ratings_with_factors(
        ratings,
        global_bias,
        note_params,
        rater_params,
        sample_label="validation",
    )
    if preds.empty:
        mae = np.nan
        mean_absolute_residual = np.nan
        median_absolute_residual = np.nan
        mse = np.nan
        rmse = np.nan
    else:
        residual = preds[mc.RATING] - preds["predHelpfulNum"]
        absolute_residual = residual.abs()
        mse = float((residual**2).mean())
        mean_absolute_residual = float(absolute_residual.mean())
        median_absolute_residual = float(absolute_residual.median())
        mae = mean_absolute_residual
        rmse = float(np.sqrt(mse))

    return {
        "model": model,
        "fit_week": cutoff.date().isoformat(),
        "prediction_window_start": cutoff.date().isoformat(),
        "prediction_window_end": prediction_end.date().isoformat(),
        "target_ratings": int(len(ratings)),
        "exact_match_predictions": int(len(preds)),
        "coverage": float(len(preds) / len(ratings)) if len(ratings) else np.nan,
        "mae": mae,
        "mean_absolute_residual": mean_absolute_residual,
        "median_absolute_residual": median_absolute_residual,
        "mse": mse,
        "rmse": rmse,
        **extra,
    }


def _estimate_user_variances_from_low_rank(
    train: pd.DataFrame,
    fit: mc.MatrixCompletionFit,
    *,
    variance_floor: float,
) -> pd.DataFrame:
    preds = mc.predict_pairs(train, fit)
    preds["residual"] = preds[mc.RATING] - preds["predHelpfulNum"]
    user_variances = (
        preds.assign(squared_residual=preds["residual"] ** 2)
        .groupby(mc.RATER_ID)
        .agg(
            n_ratings=("residual", "size"),
            residual_mean=("residual", "mean"),
            mean_squared_residual=("squared_residual", "mean"),
        )
        .reset_index()
    )
    user_variances["mean_squared_residual_floored"] = user_variances[
        "mean_squared_residual"
    ].clip(lower=variance_floor)
    user_variances["helpfulnessVariance"] = user_variances[
        "mean_squared_residual_floored"
    ]
    user_variances["inverse_variance_weight"] = (
        1.0 / user_variances["mean_squared_residual_floored"]
    )
    return user_variances


def _save_model_params(
    params_dir: Path,
    *,
    model_label: str,
    fit_week: str,
    global_bias: float,
    note_params: pd.DataFrame,
    rater_params: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    params_dir.mkdir(parents=True, exist_ok=True)
    note_params.to_csv(params_dir / f"{model_label}_note_params.csv", index=False)
    rater_params.to_csv(params_dir / f"{model_label}_rater_params.csv", index=False)
    pd.DataFrame(
        [{"model": model_label, "fit_week": fit_week, "global_bias": global_bias}]
    ).to_csv(params_dir / f"{model_label}_global_bias.csv", index=False)
    _write_json(params_dir / f"{model_label}_metadata.json", metadata)


def _format_weekly_rater_factors(
    rater_params: pd.DataFrame,
    *,
    fit_week: str,
) -> pd.DataFrame:
    required_cols = [c.raterParticipantIdKey, c.raterFactor1Key, c.raterInterceptKey]
    missing_cols = [col for col in required_cols if col not in rater_params.columns]
    if missing_cols:
        raise ValueError(f"rater_params is missing required columns: {missing_cols}")

    out = rater_params[required_cols].copy()
    out = out.rename(
        columns={
            c.raterFactor1Key: "raterFactor",
            c.raterInterceptKey: "raterIntercept",
        }
    )
    out["date"] = fit_week
    return out[[c.raterParticipantIdKey, "raterFactor", "raterIntercept", "date"]]


def _format_weekly_note_factors(
    note_params: pd.DataFrame,
    *,
    fit_week: str,
) -> pd.DataFrame:
    required_cols = [c.noteIdKey, c.noteFactor1Key, c.noteInterceptKey]
    missing_cols = [col for col in required_cols if col not in note_params.columns]
    if missing_cols:
        raise ValueError(f"note_params is missing required columns: {missing_cols}")

    out = note_params[required_cols].copy()
    out = out.rename(
        columns={
            c.noteFactor1Key: "noteFactor",
            c.noteInterceptKey: "noteIntercept",
        }
    )
    out["date"] = fit_week
    return out[[c.noteIdKey, "noteFactor", "noteIntercept", "date"]]


def _append_weekly_factors(
    weekly_factors: dict[str, dict[str, list[pd.DataFrame]]],
    *,
    model_label: str,
    fit_week: str,
    note_params: pd.DataFrame,
    rater_params: pd.DataFrame,
) -> None:
    weekly_factors[model_label]["rater"].append(
        _format_weekly_rater_factors(rater_params, fit_week=fit_week)
    )
    weekly_factors[model_label]["note"].append(
        _format_weekly_note_factors(note_params, fit_week=fit_week)
    )


def _write_weekly_factors(
    run_dir: Path,
    weekly_factors: dict[str, dict[str, list[pd.DataFrame]]],
) -> None:
    factors_dir = run_dir / "weekly_factors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    for model_label, entity_frames in weekly_factors.items():
        for entity, frames in entity_frames.items():
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(
                    factors_dir / f"{model_label}_{entity}_factors.csv",
                    index=False,
                )


def run_weekly(args: argparse.Namespace, run_dir: Path) -> pd.DataFrame:
    ratings = mc.load_ratings(Path(args.ratings_path))
    train_start = pd.Timestamp(args.train_start)
    first_cutoff = pd.Timestamp(args.first_cutoff)
    end_date = pd.Timestamp(args.end_date)
    cutoffs = list(pd.date_range(first_cutoff, end_date, freq="7D", inclusive="left"))

    rows: list[dict[str, Any]] = []
    final_artifacts: dict[str, Any] = {}
    weekly_factors: dict[str, dict[str, list[pd.DataFrame]]] = {
        "baseline_x_unweighted": {"rater": [], "note": []},
        "low_rank_variance_weighted_x": {"rater": [], "note": []},
    }

    total_cutoffs = len(cutoffs)
    for cutoff_index, cutoff in enumerate(cutoffs, start=1):
        prediction_end = min(cutoff + pd.Timedelta(days=7), end_date)
        print(
            f"[{pd.Timestamp.now()}] Starting cutoff {cutoff_index}/{total_cutoffs}: "
            f"train < {cutoff.date()}, validate "
            f"{cutoff.date()} to {prediction_end.date()}",
            flush=True,
        )
        train = ratings[
            (ratings[mc.CREATED_AT] >= train_start)
            & (ratings[mc.CREATED_AT] < cutoff)
        ].copy()
        validation = ratings[
            (ratings[mc.CREATED_AT] >= cutoff)
            & (ratings[mc.CREATED_AT] < prediction_end)
        ].copy()
        if train.empty or validation.empty:
            continue

        raw_train_ratings = len(train)
        train = train.sort_values(mc.CREATED_AT).drop_duplicates(
            subset=[mc.NOTE_ID, mc.RATER_ID],
            keep="last",
        )
        if not args.no_cn_filter:
            train = utility.filter_ratings(train, logging=args.verbose)
            train = train.sort_values(mc.CREATED_AT).drop_duplicates(
                subset=[mc.NOTE_ID, mc.RATER_ID],
                keep="last",
            )
        if train.empty:
            continue

        if args.verbose:
            print(
                f"\nCutoff {cutoff.date()}: train={len(train)} "
                f"filtered from {raw_train_ratings}, validation={len(validation)}",
                flush=True,
            )

        shared_extra = {
            "run_label": args.run_label,
            "n_raw_train_ratings": int(raw_train_ratings),
            "n_train_ratings": int(len(train)),
            "n_train_notes": int(train[mc.NOTE_ID].nunique()),
            "n_train_raters": int(train[mc.RATER_ID].nunique()),
            "cn_filter_applied": not args.no_cn_filter,
            "seed": int(args.seed),
        }

        baseline_user_variances = pd.DataFrame(
            {
                mc.RATER_ID: train[mc.RATER_ID].dropna().unique(),
                "helpfulnessVariance": 1.0,
            }
        )
        with _suppress_stdout_if_needed(args.verbose):
            baseline = run_weighted_x_matrix_factorization(
                train,
                baseline_user_variances,
                l2_lambda=c.l2_lambda,
                logging=args.verbose,
                run_name=f"{args.run_label}_baseline_{cutoff.date()}",
                return_diagnostics=False,
            )
        baseline_metrics = _summarize_predictions(
            validation,
            model="baseline_x_unweighted",
            cutoff=cutoff,
            prediction_end=prediction_end,
            global_bias=baseline.global_bias,
            note_params=baseline.note_params,
            rater_params=baseline.rater_params,
            extra={
                **shared_extra,
                "stage1_rank": np.nan,
                "stage1_lambda": np.nan,
                "stage1_lr": np.nan,
                "stage1_epochs": np.nan,
                "stage1_convergence": np.nan,
                "stage1_epochs_run": np.nan,
                "stage1_train_mae": np.nan,
                "stage1_train_mse": np.nan,
                "stage2_l2_multiplier": 1.0,
                "stage2_l2_lambda": float(c.l2_lambda),
                "variance_floor": np.nan,
                "weight_percentile": np.nan,
                "weight_percentile_strategy": np.nan,
            },
        )
        rows.append(baseline_metrics)
        print(
            f"[{pd.Timestamp.now()}] Finished baseline for {cutoff.date()}: "
            f"MAE={baseline_metrics['mae']:.6f}, "
            f"MSE={baseline_metrics['mse']:.6f}, "
            f"coverage={baseline_metrics['coverage']:.3f}",
            flush=True,
        )

        indexed = mc.make_indexed_ratings(train)
        low_rank_fit = mc.fit_matrix_completion(
            indexed,
            validation=None,
            rank=args.stage1_rank,
            lambda_reg=args.stage1_lambda,
            lr=args.stage1_lr,
            epochs=args.stage1_epochs,
            convergence=args.stage1_convergence,
            validation_patience=0,
            seed=args.seed,
            verbose=args.verbose,
        )
        user_variances = _estimate_user_variances_from_low_rank(
            train,
            low_rank_fit,
            variance_floor=args.variance_floor,
        )
        with _suppress_stdout_if_needed(args.verbose):
            two_stage = run_weighted_x_matrix_factorization(
                train,
                user_variances,
                l2_lambda=args.stage2_l2_multiplier * c.l2_lambda,
                logging=args.verbose,
                variance_floor=args.variance_floor,
                weight_percentile=args.weight_percentile,
                weight_percentile_strategy=args.weight_percentile_strategy,
                run_name=f"{args.run_label}_two_stage_{cutoff.date()}",
                return_diagnostics=False,
            )
        two_stage_metrics = _summarize_predictions(
            validation,
            model="low_rank_variance_weighted_x",
            cutoff=cutoff,
            prediction_end=prediction_end,
            global_bias=two_stage.global_bias,
            note_params=two_stage.note_params,
            rater_params=two_stage.rater_params,
            extra={
                **shared_extra,
                "stage1_rank": int(args.stage1_rank),
                "stage1_lambda": float(args.stage1_lambda),
                "stage1_lr": float(args.stage1_lr),
                "stage1_epochs": int(args.stage1_epochs),
                "stage1_convergence": float(args.stage1_convergence),
                "stage1_epochs_run": int(low_rank_fit.epochs_run),
                "stage1_train_mae": float(low_rank_fit.train_mae),
                "stage1_train_mse": float(low_rank_fit.train_mse),
                "stage2_l2_multiplier": float(args.stage2_l2_multiplier),
                "stage2_l2_lambda": float(args.stage2_l2_multiplier * c.l2_lambda),
                "variance_floor": float(args.variance_floor),
                "weight_percentile": args.weight_percentile,
                "weight_percentile_strategy": args.weight_percentile_strategy,
            },
        )
        rows.append(two_stage_metrics)
        print(
            f"[{pd.Timestamp.now()}] Finished two-stage for {cutoff.date()}: "
            f"MAE={two_stage_metrics['mae']:.6f}, "
            f"MSE={two_stage_metrics['mse']:.6f}, "
            f"coverage={two_stage_metrics['coverage']:.3f}, "
            f"stage1_epochs_run={low_rank_fit.epochs_run}",
            flush=True,
        )
        print(
            f"[{pd.Timestamp.now()}] Weekly delta for {cutoff.date()}: "
            f"MAE={two_stage_metrics['mae'] - baseline_metrics['mae']:.6f}, "
            f"MSE={two_stage_metrics['mse'] - baseline_metrics['mse']:.6f}",
            flush=True,
        )

        fit_week = cutoff.date().isoformat()
        _append_weekly_factors(
            weekly_factors,
            model_label="baseline_x_unweighted",
            fit_week=fit_week,
            note_params=baseline.note_params,
            rater_params=baseline.rater_params,
        )
        _append_weekly_factors(
            weekly_factors,
            model_label="low_rank_variance_weighted_x",
            fit_week=fit_week,
            note_params=two_stage.note_params,
            rater_params=two_stage.rater_params,
        )
        final_artifacts = {
            "fit_week": fit_week,
            "baseline": baseline,
            "two_stage": two_stage,
            "user_variances": user_variances,
        }

        if args.save_all_weekly_params:
            week_dir = run_dir / "params_by_week" / f"fit_week={fit_week}"
            _save_model_params(
                week_dir,
                model_label="baseline_x_unweighted",
                fit_week=fit_week,
                global_bias=baseline.global_bias,
                note_params=baseline.note_params,
                rater_params=baseline.rater_params,
                metadata=baseline.metadata,
            )
            _save_model_params(
                week_dir,
                model_label="low_rank_variance_weighted_x",
                fit_week=fit_week,
                global_bias=two_stage.global_bias,
                note_params=two_stage.note_params,
                rater_params=two_stage.rater_params,
                metadata=two_stage.metadata,
            )
            user_variances.to_csv(week_dir / "stage1_user_variances.csv", index=False)

    metrics = pd.DataFrame(rows)
    _write_weekly_factors(run_dir, weekly_factors)
    if final_artifacts:
        final_dir = run_dir / "final_params"
        _save_model_params(
            final_dir,
            model_label="baseline_x_unweighted",
            fit_week=final_artifacts["fit_week"],
            global_bias=final_artifacts["baseline"].global_bias,
            note_params=final_artifacts["baseline"].note_params,
            rater_params=final_artifacts["baseline"].rater_params,
            metadata=final_artifacts["baseline"].metadata,
        )
        _save_model_params(
            final_dir,
            model_label="low_rank_variance_weighted_x",
            fit_week=final_artifacts["fit_week"],
            global_bias=final_artifacts["two_stage"].global_bias,
            note_params=final_artifacts["two_stage"].note_params,
            rater_params=final_artifacts["two_stage"].rater_params,
            metadata=final_artifacts["two_stage"].metadata,
        )
        final_artifacts["user_variances"].to_csv(
            final_dir / "stage1_user_variances.csv",
            index=False,
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the best-current weekly baseline-vs-two-stage experiment and "
            "write plotting metrics plus final fitted parameters."
        )
    )
    parser.add_argument("--ratings-path", default=str(DEFAULT_RATINGS_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--first-cutoff", default="2023-06-01")
    parser.add_argument("--end-date", default="2024-06-01")
    parser.add_argument("--stage1-rank", type=int, default=2)
    parser.add_argument("--stage1-lambda", type=float, default=0.03)
    parser.add_argument("--stage1-lr", type=float, default=0.01)
    parser.add_argument("--stage1-epochs", type=int, default=1500)
    parser.add_argument("--stage1-convergence", type=float, default=1e-5)
    parser.add_argument("--stage2-l2-multiplier", type=float, default=0.5)
    parser.add_argument("--variance-floor", type=float, default=1e-3)
    parser.add_argument("--weight-percentile", type=float, default=None)
    parser.add_argument(
        "--weight-percentile-strategy",
        choices=["mean", "max"],
        default="mean",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cn-filter", action="store_true")
    parser.add_argument("--save-all-weekly-params", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.run_label is None:
        args.run_label = _default_run_label()
    if pd.Timestamp(args.first_cutoff) >= pd.Timestamp(args.end_date):
        raise ValueError("--first-cutoff must be before --end-date.")
    if args.stage1_rank < 1:
        raise ValueError("--stage1-rank must be at least 1.")
    if args.stage1_lambda < 0:
        raise ValueError("--stage1-lambda must be nonnegative.")
    if args.stage1_lr <= 0:
        raise ValueError("--stage1-lr must be positive.")
    if args.stage1_epochs < 1:
        raise ValueError("--stage1-epochs must be at least 1.")
    if args.stage2_l2_multiplier <= 0:
        raise ValueError("--stage2-l2-multiplier must be positive.")
    if args.variance_floor <= 0:
        raise ValueError("--variance-floor must be positive.")
    return args


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_root) / args.run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_label": args.run_label,
            "script": str(Path(__file__).resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "args": vars(args),
        },
    )
    metrics = run_weekly(args, run_dir)
    metrics_path = run_dir / "weekly_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print(f"Wrote metrics: {metrics_path}", flush=True)
    print(f"Wrote weekly factors: {run_dir / 'weekly_factors'}", flush=True)
    print(f"Wrote final params: {run_dir / 'final_params'}", flush=True)
    if not metrics.empty:
        summary = (
            metrics.pivot(
                index="fit_week",
                columns="model",
                values=["mean_absolute_residual", "median_absolute_residual", "mse"],
            )
            .sort_index()
        )
        print(summary.to_string(float_format=lambda x: f"{x:.6f}"), flush=True)


if __name__ == "__main__":
    main()
