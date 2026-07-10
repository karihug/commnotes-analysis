"""Self-contained helpers for `weekly_two_stage.py`.

This consolidates the pieces the original weekly-good script imported from
`fixed_x_twopass_metrics.py`, `mf_stage2.py`, `predictions.py`, and
`matrix_factorization.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import constants as c
import utility


NOTE_ID = c.noteIdKey
RATER_ID = c.raterParticipantIdKey
RATING = c.helpfulNumKey
CREATED_AT = "createdAtDate"
NOTE_INTERCEPT = c.noteInterceptKey
RATER_INTERCEPT = c.raterInterceptKey
GLOBAL_BIAS = "globalBias"
PREDICTION = "predHelpfulNum"
DEFAULT_VARIANCE_FLOOR = 1e-3


@dataclass
class IndexedRatings:
    note_id_to_index: dict[Any, int]
    rater_id_to_index: dict[Any, int]
    index_to_note_id: np.ndarray
    index_to_rater_id: np.ndarray
    row_idx: np.ndarray
    col_idx: np.ndarray
    values: np.ndarray

    @property
    def num_notes(self) -> int:
        return len(self.index_to_note_id)

    @property
    def num_raters(self) -> int:
        return len(self.index_to_rater_id)

    @property
    def num_observed(self) -> int:
        return len(self.values)


@dataclass
class IndexedEvaluationRatings:
    row_idx: np.ndarray
    col_idx: np.ndarray
    values: np.ndarray
    target_count: int

    @property
    def num_observed(self) -> int:
        return len(self.values)


@dataclass
class MatrixCompletionFit:
    rank: int
    lambda_reg: float
    lr: float
    global_bias: float
    note_factors: np.ndarray
    rater_factors: np.ndarray
    note_id_to_index: dict[Any, int]
    rater_id_to_index: dict[Any, int]
    train_rmse: float
    train_mae: float
    train_mse: float
    final_loss: float
    epochs_run: int
    best_validation_epoch: int | None
    best_validation_mae: float
    best_validation_mse: float
    best_validation_rmse: float
    validation_checks: int
    early_stopped: bool


@dataclass
class WeightedMFResult:
    global_bias: float
    note_params: pd.DataFrame
    rater_params: pd.DataFrame
    user_variances_for_mf: pd.DataFrame
    metadata: dict
    epoch_diagnostics: Optional[pd.DataFrame]


class LowRankMatrixCompletion(torch.nn.Module):
    def __init__(self, num_notes: int, num_raters: int, rank: int) -> None:
        super().__init__()
        self.note_factors = torch.nn.Embedding(num_notes, rank, sparse=False)
        self.rater_factors = torch.nn.Embedding(num_raters, rank, sparse=False)
        torch.nn.init.xavier_uniform_(self.note_factors.weight)
        torch.nn.init.xavier_uniform_(self.rater_factors.weight)

    def forward(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        return (self.note_factors(rows) * self.rater_factors(cols)).sum(dim=1)

    # def factor_penalty(self) -> torch.Tensor:
    #     return (self.note_factors.weight**2).mean() + (
    #         self.rater_factors.weight**2
    #     ).mean()

    def factor_penalty(self) -> torch.Tensor:
        return 0.5 * (
            self.note_factors.weight.pow(2).sum()
            + self.rater_factors.weight.pow(2).sum()
        )


class BiasedMatrixFactorization(torch.nn.Module):
  """Matrix factorization algorithm class."""

  def __init__(
    self, n_users: int, n_items: int, n_factors: int = 1, use_global_intercept: bool = True
  ) -> None:
    """Initialize matrix factorization model using xavier_uniform for factors
    and zeros for intercepts.

    Args:
        n_users (int): number of raters
        n_items (int): number of notes
        n_factors (int, optional): number of dimensions. Defaults to 1. Only 1 is supported.
        use_global_intercept (bool, optional): Defaults to True.
    """
    super().__init__()
    self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=False)
    self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=False)
    self.user_intercepts = torch.nn.Embedding(n_users, 1, sparse=False)
    self.item_intercepts = torch.nn.Embedding(n_items, 1, sparse=False)
    self.use_global_intercept = use_global_intercept
    self.global_intercept = torch.nn.parameter.Parameter(torch.zeros(1, 1))
    torch.nn.init.xavier_uniform_(self.user_factors.weight)
    torch.nn.init.xavier_uniform_(self.item_factors.weight)
    self.user_intercepts.weight.data.fill_(0.0)
    self.item_intercepts.weight.data.fill_(0.0)

  def forward(self, user, item):
    """Forward pass: get predicted rating for user of note (item)"""
    pred = self.user_intercepts(user) + self.item_intercepts(item)
    pred += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
    if self.use_global_intercept == True:
      pred += self.global_intercept
    return pred.squeeze()


def load_ratings(path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(path)
    if CREATED_AT in ratings.columns:
        ratings[CREATED_AT] = pd.to_datetime(ratings[CREATED_AT])
    elif c.createdAtMillisKey in ratings.columns:
        ratings[CREATED_AT] = pd.to_datetime(ratings[c.createdAtMillisKey], unit="ms")
    else:
        raise ValueError(f"ratings must contain {CREATED_AT} or {c.createdAtMillisKey}.")

    if RATING not in ratings.columns:
        if c.helpfulnessLevelKey not in ratings.columns:
            raise ValueError(
                f"ratings must contain {RATING} or {c.helpfulnessLevelKey}."
            )
        ratings[RATING] = ratings[c.helpfulnessLevelKey].map(
            {
                c.helpfulValueTsv: 1.0,
                c.somewhatHelpfulValueTsv: 0.5,
                c.notHelpfulValueTsv: 0.0,
            }
        )

    ratings = ratings.dropna(subset=[NOTE_ID, RATER_ID, RATING, CREATED_AT]).copy()
    ratings[RATING] = ratings[RATING].astype(float)
    return ratings


def prepare_train_ratings(
    ratings: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    *,
    apply_cn_filter: bool,
    verbose: bool,
) -> pd.DataFrame:
    train = ratings[
        (ratings[CREATED_AT] >= train_start) & (ratings[CREATED_AT] < train_end)
    ].copy()
    train = train.sort_values(CREATED_AT)
    train = train.drop_duplicates(subset=[NOTE_ID, RATER_ID], keep="last")

    if apply_cn_filter:
        train = utility.filter_ratings(train, logging=verbose)
        train = train.sort_values(CREATED_AT)
        train = train.drop_duplicates(subset=[NOTE_ID, RATER_ID], keep="last")

    if train.empty:
        raise ValueError("No training ratings remain after filtering.")
    return train


def make_indexed_ratings(train_ratings: pd.DataFrame) -> IndexedRatings:
    index_to_note_id = train_ratings[NOTE_ID].drop_duplicates().to_numpy()
    index_to_rater_id = train_ratings[RATER_ID].drop_duplicates().to_numpy()
    note_id_to_index = {note_id: idx for idx, note_id in enumerate(index_to_note_id)}
    rater_id_to_index = {
        rater_id: idx for idx, rater_id in enumerate(index_to_rater_id)
    }
    return IndexedRatings(
        note_id_to_index=note_id_to_index,
        rater_id_to_index=rater_id_to_index,
        index_to_note_id=index_to_note_id,
        index_to_rater_id=index_to_rater_id,
        row_idx=train_ratings[NOTE_ID].map(note_id_to_index).to_numpy(dtype=np.int64),
        col_idx=train_ratings[RATER_ID].map(rater_id_to_index).to_numpy(
            dtype=np.int64
        ),
        values=train_ratings[RATING].to_numpy(dtype=np.float32),
    )


def make_indexed_evaluation_ratings(
    ratings: pd.DataFrame,
    indexed_train: IndexedRatings,
) -> IndexedEvaluationRatings:
    eval_ratings = ratings.copy()
    eval_ratings["_note_index"] = eval_ratings[NOTE_ID].map(
        indexed_train.note_id_to_index
    )
    eval_ratings["_rater_index"] = eval_ratings[RATER_ID].map(
        indexed_train.rater_id_to_index
    )
    eval_ratings = eval_ratings.dropna(subset=["_note_index", "_rater_index"])
    return IndexedEvaluationRatings(
        row_idx=eval_ratings["_note_index"].to_numpy(dtype=np.int64),
        col_idx=eval_ratings["_rater_index"].to_numpy(dtype=np.int64),
        values=eval_ratings[RATING].to_numpy(dtype=np.float32),
        target_count=len(ratings),
    )


def _error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "mae": np.nan,
            "mean_absolute_residual": np.nan,
            "median_absolute_residual": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
        }
    residual = y_true - y_pred
    absolute_residual = np.abs(residual)
    mse = float(np.mean(residual**2))
    return {
        "mae": float(np.mean(absolute_residual)),
        "mean_absolute_residual": float(np.mean(absolute_residual)),
        "median_absolute_residual": float(np.median(absolute_residual)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def fit_matrix_completion(
    indexed: IndexedRatings,
    *,
    validation: IndexedEvaluationRatings | None,
    rank: int,
    lambda_reg: float,
    lr: float,
    epochs: int,
    convergence: float,
    validation_patience: int,
    seed: int,
    verbose: bool,
) -> MatrixCompletionFit:
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rows = torch.LongTensor(indexed.row_idx).to(device)
    cols = torch.LongTensor(indexed.col_idx).to(device)
    values = torch.FloatTensor(indexed.values).to(device)
    global_bias = float(indexed.values.mean())
    centered_values = values - global_bias
    if validation is not None and validation.num_observed:
        validation_rows = torch.LongTensor(validation.row_idx).to(device)
        validation_cols = torch.LongTensor(validation.col_idx).to(device)
        validation_values = torch.FloatTensor(validation.values).to(device)
    else:
        validation_rows = None
        validation_cols = None
        validation_values = None

    model = LowRankMatrixCompletion(
        indexed.num_notes,
        indexed.num_raters,
        rank,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    previous_loss = np.inf
    final_loss = np.nan
    best_validation_epoch = None
    best_validation_metrics = {"mae": np.nan, "mse": np.nan, "rmse": np.nan}
    best_validation_state = None
    validation_checks = 0
    epochs_since_validation_improvement = 0
    early_stopped = False

    def validation_metrics_for_current_model() -> dict[str, float]:
        if validation_rows is None or validation_cols is None or validation_values is None:
            return {"mae": np.nan, "mse": np.nan, "rmse": np.nan}
        with torch.no_grad():
            validation_pred = global_bias + model(validation_rows, validation_cols)
            residual = validation_values - validation_pred
            mse = torch.mean(residual**2)
            mae = torch.mean(torch.abs(residual))
        return {
            "mae": float(mae.detach().cpu().item()),
            "mse": float(mse.detach().cpu().item()),
            "rmse": float(torch.sqrt(mse).detach().cpu().item()),
        }

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        centered_pred = model(rows, cols)

        # variational characterization of NN loss
        fit_loss = 0.5 * torch.mean((centered_pred - centered_values) ** 2)
        reg_loss = lambda_reg * model.factor_penalty() / len(values)
        loss = fit_loss + reg_loss
        final_loss = float(loss.detach().cpu().item())
        current_validation_metrics = validation_metrics_for_current_model()
        validation_checks += int(np.isfinite(current_validation_metrics["mse"]))
        if (
            np.isfinite(current_validation_metrics["mse"])
            and (
                best_validation_epoch is None
                or current_validation_metrics["mse"] < best_validation_metrics["mse"]
            )
        ):
            best_validation_epoch = epoch
            best_validation_metrics = current_validation_metrics
            best_validation_state = {
                name: param.detach().cpu().clone()
                for name, param in model.state_dict().items()
            }
            epochs_since_validation_improvement = 0
        elif np.isfinite(current_validation_metrics["mse"]):
            epochs_since_validation_improvement += 1

        if verbose and (epoch == 0 or epoch % 50 == 0):
            train_rmse = torch.sqrt(fit_loss).detach().cpu().item()
            validation_rmse = current_validation_metrics["rmse"]
            print(
                f"rank={rank} lambda={lambda_reg} lr={lr} epoch={epoch} "
                f"loss={final_loss:.6f} fit_rmse={train_rmse:.6f} "
                f"validation_rmse={validation_rmse:.6f}"
            )

        if (
            validation_patience > 0
            and np.isfinite(current_validation_metrics["mse"])
            and epochs_since_validation_improvement >= validation_patience
        ):
            early_stopped = True
            break
        if epoch > 0 and abs(previous_loss - final_loss) <= convergence:
            break
        previous_loss = final_loss
        if epoch == epochs:
            break
        loss.backward()
        optimizer.step()

    if best_validation_state is not None:
        model.load_state_dict(
            {
                name: tensor.to(device)
                for name, tensor in best_validation_state.items()
            }
        )

    with torch.no_grad():
        train_pred = (global_bias + model(rows, cols)).detach().cpu().numpy()
    train_metrics = _error_metrics(indexed.values.astype(float), train_pred)

    return MatrixCompletionFit(
        rank=rank,
        lambda_reg=lambda_reg,
        lr=lr,
        global_bias=global_bias,
        note_factors=model.note_factors.weight.detach().cpu().numpy(),
        rater_factors=model.rater_factors.weight.detach().cpu().numpy(),
        note_id_to_index=indexed.note_id_to_index,
        rater_id_to_index=indexed.rater_id_to_index,
        train_rmse=train_metrics["rmse"],
        train_mae=train_metrics["mae"],
        train_mse=train_metrics["mse"],
        final_loss=final_loss,
        epochs_run=epoch,
        best_validation_epoch=best_validation_epoch,
        best_validation_mae=best_validation_metrics["mae"],
        best_validation_mse=best_validation_metrics["mse"],
        best_validation_rmse=best_validation_metrics["rmse"],
        validation_checks=validation_checks,
        early_stopped=early_stopped,
    )


def predict_pairs(ratings: pd.DataFrame, fit: MatrixCompletionFit) -> pd.DataFrame:
    pred = ratings.copy()
    pred["_note_index"] = pred[NOTE_ID].map(fit.note_id_to_index)
    pred["_rater_index"] = pred[RATER_ID].map(fit.rater_id_to_index)
    pred = pred.dropna(subset=["_note_index", "_rater_index"]).copy()
    if pred.empty:
        pred["predHelpfulNum"] = []
        return pred.drop(columns=["_note_index", "_rater_index"])

    row_idx = pred["_note_index"].to_numpy(dtype=np.int64)
    col_idx = pred["_rater_index"].to_numpy(dtype=np.int64)
    pred["predHelpfulNum"] = fit.global_bias + np.sum(
        fit.note_factors[row_idx] * fit.rater_factors[col_idx],
        axis=1,
    )
    return pred.drop(columns=["_note_index", "_rater_index"])


def summarize_prediction_frame(
    ratings: pd.DataFrame,
    fit: MatrixCompletionFit,
    *,
    sample: str,
    horizon_week: int | None,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    clip_predictions: bool,
) -> dict[str, Any]:
    preds = predict_pairs(ratings, fit)
    if preds.empty:
        metrics = {
            "mae": np.nan,
            "mean_absolute_residual": np.nan,
            "median_absolute_residual": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
        }
    else:
        y_hat = preds["predHelpfulNum"].to_numpy(dtype=float)
        if clip_predictions:
            y_hat = np.clip(y_hat, 0.0, 1.0)
        metrics = _error_metrics(preds[RATING].to_numpy(dtype=float), y_hat)

    return {
        "sample": sample,
        "horizon_week": horizon_week,
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "target_ratings": int(len(ratings)),
        "exact_match_predictions": int(len(preds)),
        "coverage": float(len(preds) / len(ratings)) if len(ratings) else np.nan,
        **metrics,
    }


def validation_windows(
    ratings: pd.DataFrame,
    train_end: pd.Timestamp,
    max_horizon_weeks: int,
) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    windows = []
    for horizon_week in range(1, max_horizon_weeks + 1):
        window_start = train_end + pd.Timedelta(days=7 * (horizon_week - 1))
        window_end = window_start + pd.Timedelta(days=7)
        window = ratings[
            (ratings[CREATED_AT] >= window_start)
            & (ratings[CREATED_AT] < window_end)
        ].copy()
        windows.append((horizon_week, window_start, window_end, window))
    return windows


def evaluate_fit(
    fit: MatrixCompletionFit,
    train: pd.DataFrame,
    windows: list[tuple[int, pd.Timestamp, pd.Timestamp, pd.DataFrame]],
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    clip_predictions: bool,
) -> pd.DataFrame:
    rows = [
        summarize_prediction_frame(
            train,
            fit,
            sample="in_sample",
            horizon_week=None,
            window_start=train_start,
            window_end=train_end,
            clip_predictions=clip_predictions,
        )
    ]
    for horizon_week, window_start, window_end, ratings in windows:
        rows.append(
            summarize_prediction_frame(
                ratings,
                fit,
                sample="validation",
                horizon_week=horizon_week,
                window_start=window_start,
                window_end=window_end,
                clip_predictions=clip_predictions,
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.insert(0, "rank", fit.rank)
    metrics.insert(1, "lambda_reg", fit.lambda_reg)
    metrics.insert(2, "lr", fit.lr)
    metrics.insert(3, "global_bias", fit.global_bias)
    metrics.insert(4, "epochs_run", fit.epochs_run)
    metrics.insert(5, "final_loss", fit.final_loss)
    metrics.insert(6, "early_stopped", fit.early_stopped)
    metrics.insert(7, "best_validation_epoch", fit.best_validation_epoch)
    metrics.insert(8, "best_validation_mae", fit.best_validation_mae)
    metrics.insert(9, "best_validation_mse", fit.best_validation_mse)
    metrics.insert(10, "best_validation_rmse", fit.best_validation_rmse)
    metrics.insert(11, "validation_checks", fit.validation_checks)
    metrics.insert(12, "train_mae_from_fit", fit.train_mae)
    metrics.insert(13, "train_mse_from_fit", fit.train_mse)
    metrics.insert(14, "train_rmse_from_fit", fit.train_rmse)
    return metrics


def _extract_global_bias(global_bias) -> float:
    if global_bias is None:
        return np.nan

    if isinstance(global_bias, pd.DataFrame):
        if GLOBAL_BIAS not in global_bias.columns:
            raise ValueError(f"global_bias dataframe must contain {GLOBAL_BIAS!r}.")
        return float(global_bias[GLOBAL_BIAS].iloc[0])

    if isinstance(global_bias, pd.Series):
        if GLOBAL_BIAS in global_bias.index:
            return float(global_bias[GLOBAL_BIAS])
        return float(global_bias.iloc[0])

    value = global_bias
    if isinstance(value, (list, tuple)):
        try:
            value = value[0][0]
        except Exception:
            value = value[0]

    if hasattr(value, "detach"):
        return float(value.detach().cpu().numpy().reshape(-1)[0])

    return float(np.asarray(value).reshape(-1)[0])


def _factor_sort_key(column: str) -> int:
    match = re.search(r"(\d+)$", column)
    return int(match.group(1)) if match else 0


def _available_factor_pairs(
    note_params: pd.DataFrame,
    rater_params: pd.DataFrame,
) -> list[tuple[str, str]]:
    note_factor_cols = sorted(
        [col for col in note_params.columns if col.startswith("noteFactor")],
        key=_factor_sort_key,
    )
    factor_pairs = []
    for note_col in note_factor_cols:
        rater_col = note_col.replace("noteFactor", "raterFactor")
        if rater_col in rater_params.columns:
            factor_pairs.append((note_col, rater_col))
    return factor_pairs


def predict_ratings_with_factors(
    ratings_df: pd.DataFrame,
    global_bias,
    note_params: pd.DataFrame,
    rater_params: pd.DataFrame,
    *,
    sample_label: Optional[str] = None,
) -> pd.DataFrame:
    """Predict ratings where both note and rater parameters are available.

    Prediction formula:
        global_bias + note_intercept + rater_intercept
        + sum_k note_factor_k * rater_factor_k
    """
    required_rating_cols = [NOTE_ID, RATER_ID]
    missing_rating_cols = [
        col for col in required_rating_cols if col not in ratings_df.columns
    ]
    if missing_rating_cols:
        raise ValueError(f"ratings_df is missing required columns: {missing_rating_cols}")

    required_note_cols = [NOTE_ID, NOTE_INTERCEPT]
    required_rater_cols = [RATER_ID, RATER_INTERCEPT]
    missing_note_cols = [col for col in required_note_cols if col not in note_params.columns]
    missing_rater_cols = [col for col in required_rater_cols if col not in rater_params.columns]
    if missing_note_cols:
        raise ValueError(f"note_params is missing required columns: {missing_note_cols}")
    if missing_rater_cols:
        raise ValueError(f"rater_params is missing required columns: {missing_rater_cols}")

    bias = _extract_global_bias(global_bias)
    factor_pairs = _available_factor_pairs(note_params, rater_params)

    note_keep_cols = [NOTE_ID, NOTE_INTERCEPT] + [
        note_col for note_col, _ in factor_pairs
    ]
    rater_keep_cols = [RATER_ID, RATER_INTERCEPT] + [
        rater_col for _, rater_col in factor_pairs
    ]

    pred_df = ratings_df.copy()
    pred_df = pred_df.merge(
        note_params[note_keep_cols],
        on=NOTE_ID,
        how="inner",
    )
    pred_df = pred_df.merge(
        rater_params[rater_keep_cols],
        on=RATER_ID,
        how="inner",
    )

    pred = bias + pred_df[NOTE_INTERCEPT] + pred_df[RATER_INTERCEPT]
    for note_col, rater_col in factor_pairs:
        pred = pred + pred_df[note_col] * pred_df[rater_col]

    pred_df[PREDICTION] = pred
    if sample_label is not None:
        pred_df["sample"] = sample_label

    if RATING in pred_df.columns:
        pred_df["residual"] = pred_df[RATING] - pred_df[PREDICTION]
        pred_df["squared_error"] = pred_df["residual"] ** 2
        pred_df["absolute_error"] = pred_df["residual"].abs()

    return pred_df


def predict_in_sample_and_out_of_sample(
    train_ratings_df: pd.DataFrame,
    out_of_sample_ratings_df: pd.DataFrame,
    global_bias,
    note_params: pd.DataFrame,
    rater_params: pd.DataFrame,
) -> pd.DataFrame:
    """Predict train and validation/test ratings with an explicit sample label."""
    in_sample = predict_ratings_with_factors(
        train_ratings_df,
        global_bias,
        note_params,
        rater_params,
        sample_label="in_sample",
    )
    out_of_sample = predict_ratings_with_factors(
        out_of_sample_ratings_df,
        global_bias,
        note_params,
        rater_params,
        sample_label="out_of_sample",
    )
    return pd.concat([in_sample, out_of_sample], ignore_index=True)


def prepare_user_variances_for_x_mf(
    user_variances: pd.DataFrame,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    weight_percentile: Optional[float] = None,
    weight_percentile_strategy: str = "mean",
) -> pd.DataFrame:
    """Return the user variance frame expected by X's run_mf function."""
    if RATER_ID not in user_variances.columns:
        raise ValueError(f"user_variances must contain {RATER_ID!r}.")
    if weight_percentile is not None and (
        weight_percentile <= 0 or weight_percentile >= 100
    ):
        raise ValueError("weight_percentile must be between 0 and 100.")
    if weight_percentile_strategy not in {"mean", "max"}:
        raise ValueError("weight_percentile_strategy must be 'mean' or 'max'.")

    out = user_variances.copy()
    if "helpfulnessVariance" in out.columns:
        variance = out["helpfulnessVariance"]
    elif "mean_squared_residual_floored" in out.columns:
        variance = out["mean_squared_residual_floored"]
    elif "mean_squared_residual" in out.columns:
        variance = out["mean_squared_residual"]
    elif "residual_variance_floored" in out.columns:
        variance = out["residual_variance_floored"]
    elif "residual_variance" in out.columns:
        variance = out["residual_variance"]
    elif "inverse_variance_weight" in out.columns:
        variance = 1.0 / out["inverse_variance_weight"]
    else:
        raise ValueError(
            "user_variances must contain one of: "
            "'helpfulnessVariance', 'mean_squared_residual_floored', "
            "'mean_squared_residual', 'residual_variance_floored', "
            "'residual_variance', or 'inverse_variance_weight'."
        )

    out = out[[RATER_ID]].copy()
    if "n_ratings" in user_variances.columns:
        out["nRatingsForWeightScale"] = pd.to_numeric(
            user_variances["n_ratings"],
            errors="coerce",
        )
    elif "numRatingsForVariance" in user_variances.columns:
        out["nRatingsForWeightScale"] = pd.to_numeric(
            user_variances["numRatingsForVariance"],
            errors="coerce",
        )
    else:
        out["nRatingsForWeightScale"] = 1.0

    out["helpfulnessVarianceRaw"] = pd.to_numeric(variance, errors="coerce")
    out = out.dropna(subset=["helpfulnessVarianceRaw"])
    out["nRatingsForWeightScale"] = out["nRatingsForWeightScale"].fillna(1.0)
    out["helpfulnessVarianceRaw"] = out["helpfulnessVarianceRaw"].clip(
        lower=variance_floor
    )
    out["inverseVarianceWeightRaw"] = 1.0 / out["helpfulnessVarianceRaw"]
    out["inverseVarianceWeight"] = out["inverseVarianceWeightRaw"]

    if weight_percentile is not None:
        threshold = out["inverseVarianceWeightRaw"].quantile(
            weight_percentile / 100.0
        )
        mean_weight = out["inverseVarianceWeightRaw"].mean()
        large_weight_mask = out["inverseVarianceWeightRaw"] > threshold
        if weight_percentile_strategy == "mean":
            replacement_weight = mean_weight
        else:
            replacement_weight = threshold
        out.loc[large_weight_mask, "inverseVarianceWeight"] = replacement_weight
        out["largeWeightThreshold"] = threshold
        out["largeWeightReplacementMean"] = mean_weight
        out["largeWeightReplacementValue"] = replacement_weight
        out["largeWeightAboveThreshold"] = large_weight_mask
        out["largeWeightRevertedToMean"] = (
            large_weight_mask if weight_percentile_strategy == "mean" else False
        )
        out["largeWeightClippedToThreshold"] = (
            large_weight_mask if weight_percentile_strategy == "max" else False
        )
        out["weightPercentileStrategy"] = weight_percentile_strategy
    else:
        out["largeWeightThreshold"] = np.nan
        out["largeWeightReplacementMean"] = np.nan
        out["largeWeightReplacementValue"] = np.nan
        out["largeWeightAboveThreshold"] = False
        out["largeWeightRevertedToMean"] = False
        out["largeWeightClippedToThreshold"] = False
        out["weightPercentileStrategy"] = weight_percentile_strategy

    weighted_sum = (
        out["inverseVarianceWeight"] * out["nRatingsForWeightScale"]
    ).sum()
    sample_count = out["nRatingsForWeightScale"].sum()
    if not np.isfinite(weighted_sum) or weighted_sum <= 0:
        raise ValueError("Cannot normalize weights because their weighted sum is invalid.")
    weight_normalization_scale = sample_count / weighted_sum
    out["inverseVarianceWeightUnnormalized"] = out["inverseVarianceWeight"]
    out["inverseVarianceWeight"] = (
        out["inverseVarianceWeight"] * weight_normalization_scale
    )
    out["weightNormalizationScale"] = weight_normalization_scale

    out["helpfulnessVariance"] = 1.0 / out["inverseVarianceWeight"]
    out = out.drop_duplicates(subset=[RATER_ID], keep="last")
    return out


def _prepare_ratings_for_x_mf(
    ratings_df: pd.DataFrame,
    user_variances_for_mf: pd.DataFrame,
    duplicate_policy: str,
) -> tuple[pd.DataFrame, dict]:
    required_cols = [NOTE_ID, RATER_ID, RATING]
    missing_cols = [col for col in required_cols if col not in ratings_df.columns]
    if missing_cols:
        raise ValueError(f"ratings_df is missing required columns: {missing_cols}")
    if duplicate_policy != "take_most_recent":
        raise ValueError("Only duplicate_policy='take_most_recent' is supported.")

    cols = [NOTE_ID, RATER_ID, RATING]
    if CREATED_AT in ratings_df.columns:
        cols.append(CREATED_AT)

    clean_df = ratings_df[cols].dropna(subset=[NOTE_ID, RATER_ID, RATING]).copy()
    clean_df[RATING] = clean_df[RATING].astype(float)

    if CREATED_AT in clean_df.columns:
        clean_df[CREATED_AT] = pd.to_datetime(clean_df[CREATED_AT])
        clean_df = clean_df.sort_values(CREATED_AT)

    clean_df = clean_df.drop_duplicates(subset=[NOTE_ID, RATER_ID], keep="last")
    n_after_duplicates = len(clean_df)

    covered_raters = set(user_variances_for_mf[RATER_ID])
    clean_df = clean_df[clean_df[RATER_ID].isin(covered_raters)].copy()
    if clean_df.empty:
        raise ValueError("No ratings remain after keeping raters with user variances.")

    metadata = {
        "n_input_ratings": int(len(ratings_df)),
        "n_after_dropna": int(ratings_df.dropna(subset=[NOTE_ID, RATER_ID, RATING]).shape[0]),
        "n_after_duplicates": int(n_after_duplicates),
        "n_after_user_variance_filter": int(len(clean_df)),
        "n_dropped_missing_user_variance": int(n_after_duplicates - len(clean_df)),
        "n_notes": int(clean_df[NOTE_ID].nunique()),
        "n_raters": int(clean_df[RATER_ID].nunique()),
    }

    if CREATED_AT in clean_df.columns:
        training_start = clean_df[CREATED_AT].min()
        training_end = clean_df[CREATED_AT].max()
        last_calendar_week_start = training_end - pd.to_timedelta(
            training_end.dayofweek,
            unit="d",
        )
        last_calendar_week_end = last_calendar_week_start + pd.Timedelta(days=6)
        last_7_days_start = training_end - pd.Timedelta(days=6)

        metadata.update(
            {
                "training_start_date": training_start,
                "training_end_date": training_end,
                "last_calendar_week_start": last_calendar_week_start,
                "last_calendar_week_end": last_calendar_week_end,
                "last_7_days_start": last_7_days_start,
                "last_7_days_end": training_end,
                "last_7_days_n_ratings": int(
                    clean_df[CREATED_AT].between(last_7_days_start, training_end).sum()
                ),
            }
        )

    return clean_df, metadata


def run_mf(
  ratings: pd.DataFrame,
  l2_lambda: float,
  l2_intercept_multiplier: float,
  numFactors: int,
  epochs: int,
  useGlobalIntercept: bool,
  runName: str = "prod",
  logging: bool = True,
  flipFactorsForIdentification: bool = True,
  noteInit: pd.DataFrame = None,
  userInit: pd.DataFrame = None,
  user_variances: pd.DataFrame = None,
  return_diagnostics: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
  """Train matrix factorization model.

  See https://twitter.github.io/birdwatch/ranking-notes/#matrix-factorization

  Args:
      ratings (pd.DataFrame): pre-filtered ratings to train on
      l2_lambda (float): regularization for factors
      l2_intercept_multiplier (float): how much extra to regularize intercepts
      numFactors (int): number of dimensions (only 1 is implemented)
      epochs (int): number of rounds of training
      useGlobalIntercept (bool): whether to fit global intercept parameter
      runName (str, optional): name. Defaults to "prod".
      logging (bool, optional): debug output. Defaults to True.
      flipFactorsForIdentification (bool, optional): Default to True.
      user_variances (pd.DataFrame, optional): DataFrame with columns [raterParticipantIdKey, variance]
        for weighted loss. If provided, loss is weighted by 1/variance per user. Defaults to None.
      return_diagnostics (bool, optional): If True, return a fourth element with per-epoch losses.

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, float]:
        noteParams: contains one row per note, including noteId and learned note parameters
        raterParams: contains one row per rating, including raterId and learned rater parameters
        globalIntercept: learned global intercept parameter
  """
  assert numFactors == 1
  # We are extracting only the subset of note data from the ratings data frame that is needed to
  # run matrix factorization. This avoids accidentally loosing data through `dropna`.
  noteData = ratings[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]]
  assert not pd.isna(noteData).values.any(), "noteData must note contain nan values"

  noteIdMap = (
    pd.DataFrame(noteData[c.noteIdKey].unique())
    .reset_index()
    .set_index(0)
    .reset_index()
    .rename(columns={0: c.noteIdKey, "index": c.noteIndexKey})
  )
  raterIdMap = (
    pd.DataFrame(noteData[c.raterParticipantIdKey].unique())
    .reset_index()
    .set_index(0)
    .reset_index()
    .rename(columns={0: c.raterParticipantIdKey, "index": c.raterIndexKey})
  )

  noteRatingIds = noteData.merge(noteIdMap, on=c.noteIdKey)
  noteRatingIds = noteRatingIds.merge(raterIdMap, on=c.raterParticipantIdKey)

  # Step 3: Merge user variances if provided
  if user_variances is not None:
    noteRatingIds = noteRatingIds.merge(
      user_variances,
      on=c.raterParticipantIdKey,
      how='left'
    )
    # Ensure no NaN variances
    assert not noteRatingIds['helpfulnessVariance'].isna().any(), "user_variances must cover all raters in ratings"
    noteRatingIds['helpfulnessVariance'] = noteRatingIds['helpfulnessVariance'].clip(lower=1e-4)

  n_users = noteRatingIds[c.raterIndexKey].nunique()
  n_items = noteRatingIds[c.noteIndexKey].nunique()
  if logging:
    print("------------------")
    print(f"Users: {n_users}, Notes: {n_items}")

  criterion = torch.nn.MSELoss()

  l2_lambda_intercept = l2_lambda * l2_intercept_multiplier

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  rating = torch.FloatTensor(noteRatingIds[c.helpfulNumKey].values).to(device)
  row = torch.LongTensor(noteRatingIds[c.raterIndexKey].values).to(device)
  col = torch.LongTensor(noteRatingIds[c.noteIndexKey].values).to(device)

  # Step 4: Convert user variances to torch tensor for weighted loss
  use_weighted_loss = user_variances is not None
  if use_weighted_loss:
    variance_weights = torch.FloatTensor(noteRatingIds['helpfulnessVariance'].values).to(device)
    if logging:
      print("Using weighted loss with user variances")

  mf_model = BiasedMatrixFactorization(
    n_users, n_items, use_global_intercept=useGlobalIntercept, n_factors=numFactors
  )

  if noteInit is not None:
    print("initializing notes")
    noteInit = noteIdMap.merge(noteInit, on=c.noteIdKey, how="left")
    note_intercept_mask = noteInit[c.noteInterceptKey].notna().values
    note_factor_mask = noteInit[c.noteFactor1Key].notna().values
    with torch.no_grad():
      if note_intercept_mask.any():
        mf_model.item_intercepts.weight.data[note_intercept_mask, 0] = torch.tensor(
          noteInit.loc[note_intercept_mask, c.noteInterceptKey].values,
          dtype=torch.float32,
        )
      if note_factor_mask.any():
        mf_model.item_factors.weight.data[note_factor_mask, 0] = torch.tensor(
          noteInit.loc[note_factor_mask, c.noteFactor1Key].values,
          dtype=torch.float32,
        )

  if userInit is not None:
    print("initializing users")
    userInit = raterIdMap.merge(userInit, on=c.raterParticipantIdKey, how="left")
    user_intercept_mask = userInit[c.raterInterceptKey].notna().values
    user_factor_mask = userInit[c.raterFactor1Key].notna().values
    with torch.no_grad():
      if user_intercept_mask.any():
        mf_model.user_intercepts.weight.data[user_intercept_mask, 0] = torch.tensor(
          userInit.loc[user_intercept_mask, c.raterInterceptKey].values,
          dtype=torch.float32,
        )
      if user_factor_mask.any():
        mf_model.user_factors.weight.data[user_factor_mask, 0] = torch.tensor(
          userInit.loc[user_factor_mask, c.raterFactor1Key].values,
          dtype=torch.float32,
        )

  mf_model.to(device)

  if (noteInit is not None) and (userInit is not None):
    optimizer = torch.optim.Adam(
      mf_model.parameters(), lr=c.initLearningRate
    )  # smaller learning rate
  else:
    optimizer = torch.optim.Adam(mf_model.parameters(), lr=c.noInitLearningRate)  # learning rate

  def print_loss():
    y_pred = mf_model(row, col)
    train_loss = criterion(y_pred, rating)

    if logging:
      print("epoch", epoch, loss.item())
      print("TRAIN FIT LOSS: ", train_loss.item())

  epoch_diagnostics = []

  def append_epoch_diagnostics(epoch_value, y_pred, fit_loss, reg_loss, total_loss):
    with torch.no_grad():
      train_mse = criterion(y_pred, rating)
      train_mae = torch.abs(y_pred - rating).mean()
      if use_weighted_loss:
        weighted_fit_loss = ((y_pred - rating) ** 2 / variance_weights).mean()
      else:
        weighted_fit_loss = train_mse
      epoch_diagnostics.append(
        {
          "runName": runName,
          "epoch": int(epoch_value),
          "trainMSE": float(train_mse.detach().cpu().item()),
          "trainRMSE": float(torch.sqrt(train_mse).detach().cpu().item()),
          "trainMAE": float(train_mae.detach().cpu().item()),
          "weightedTrainLoss": float(weighted_fit_loss.detach().cpu().item()),
          "fitLoss": float(fit_loss.detach().cpu().item()),
          "regularizationLoss": float(reg_loss.detach().cpu().item()),
          "regularizedLoss": float(total_loss.detach().cpu().item()),
          "useWeightedLoss": bool(use_weighted_loss),
          "l2Lambda": float(l2_lambda),
          "l2InterceptMultiplier": float(l2_intercept_multiplier),
        }
      )

  prev_loss = 1e10

  y_pred = mf_model(row, col)

  # Compute loss (weighted or standard)
  if use_weighted_loss:
    loss = ((y_pred - rating) ** 2 / variance_weights).mean()
  else:
    loss = criterion(y_pred, rating)

  l2_reg_loss = torch.tensor(0.0).to(device)

  for name, param in mf_model.named_parameters():
    if "intercept" in name:
      l2_reg_loss += l2_lambda_intercept * (param**2).mean()
    else:
      l2_reg_loss += l2_lambda * (param**2).mean()

  loss += l2_reg_loss

  epoch = 0
  append_epoch_diagnostics(epoch, y_pred, loss - l2_reg_loss, l2_reg_loss, loss)

  # kh added max epochs
  while abs(prev_loss - loss.item()) > c.convergence and epoch < epochs:

    prev_loss = loss.item()

    # Backpropagate
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Set gradients to zero
    optimizer.zero_grad()

    # Predict and calculate loss
    y_pred = mf_model(row, col)

    # Compute loss (weighted or standard)
    if use_weighted_loss:
      loss = ((y_pred - rating) ** 2 / variance_weights).mean()
    else:
      loss = criterion(y_pred, rating)

    l2_reg_loss = torch.tensor(0.0).to(device)

    for name, param in mf_model.named_parameters():
      if "intercept" in name:
        l2_reg_loss += l2_lambda_intercept * (param**2).mean()
      else:
        l2_reg_loss += l2_lambda * (param**2).mean()

    loss += l2_reg_loss
    append_epoch_diagnostics(epoch + 1, y_pred, loss - l2_reg_loss, l2_reg_loss, loss)

    if epoch % 50 == 0:
      print_loss()

    epoch += 1

  print("Num epochs:", epoch)
  print_loss()

  assert mf_model.item_factors.weight.data.cpu().numpy().shape[0] == noteIdMap.shape[0]

  noteIdMap[c.noteFactor1Key] = mf_model.item_factors.weight.data.cpu().numpy()[:, 0]
  raterIdMap[c.raterFactor1Key] = mf_model.user_factors.weight.data.cpu().numpy()[:, 0]
  noteIdMap[c.noteInterceptKey] = mf_model.item_intercepts.weight.data.cpu().numpy()[:, 0]
  raterIdMap[c.raterInterceptKey] = mf_model.user_intercepts.weight.data.cpu().numpy()[:, 0]

  globalIntercept = None
  if useGlobalIntercept:
    globalIntercept = mf_model.global_intercept

  if flipFactorsForIdentification:
    noteIdMap, raterIdMap = flip_factors_for_identification(noteIdMap, raterIdMap)

  if return_diagnostics:
    return noteIdMap, raterIdMap, globalIntercept, pd.DataFrame(epoch_diagnostics)

  return noteIdMap, raterIdMap, globalIntercept


def flip_factors_for_identification(
  noteParams: pd.DataFrame, raterParams: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Flip factors if needed, so that the larger group of raters gets a negative factor1

  Args:
      noteParams (pd.DataFrame): note params
      raterParams (pd.DataFrame): rater params

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: noteParams, raterParams
  """
  raterFactors = raterParams.loc[~pd.isna(raterParams["raterFactor1"]), "raterFactor1"]
  propNegativeRaterFactors = (raterFactors < 0).sum() / (raterFactors != 0).sum()

  if propNegativeRaterFactors < 0.5:
    # Flip all factors, on notes and raters
    noteParams["noteFactor1"] = noteParams["noteFactor1"] * -1
    raterParams["raterFactor1"] = raterParams["raterFactor1"] * -1

  raterFactors = raterParams.loc[~pd.isna(raterParams["raterFactor1"]), "raterFactor1"]
  propNegativeRaterFactors = (raterFactors < 0).sum() / (raterFactors != 0).sum()
  assert propNegativeRaterFactors >= 0.5

  return noteParams, raterParams


def run_weighted_x_matrix_factorization(
    ratings_df: pd.DataFrame,
    user_variances: pd.DataFrame,
    l2_lambda: float = c.l2_lambda,
    l2_intercept_multiplier: float = c.l2_intercept_multiplier,
    num_factors: int = c.numFactors,
    epochs: int = c.epochs,
    use_global_intercept: bool = c.useGlobalIntercept,
    run_name: str = "two_stage_weighted_x_mf",
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    weight_percentile: Optional[float] = None,
    weight_percentile_strategy: str = "mean",
    duplicate_policy: str = "take_most_recent",
    logging: bool = True,
    flip_factors_for_identification: bool = True,
    note_init: Optional[pd.DataFrame] = None,
    user_init: Optional[pd.DataFrame] = None,
    return_diagnostics: bool = True,
) -> WeightedMFResult:
    """Run X's existing weighted Community Notes matrix factorization."""
    user_variances_for_mf = prepare_user_variances_for_x_mf(
        user_variances,
        variance_floor=variance_floor,
        weight_percentile=weight_percentile,
        weight_percentile_strategy=weight_percentile_strategy,
    )
    ratings_for_mf, metadata = _prepare_ratings_for_x_mf(
        ratings_df,
        user_variances_for_mf,
        duplicate_policy=duplicate_policy,
    )

    result = run_mf(
        ratings_for_mf,
        l2_lambda,
        l2_intercept_multiplier,
        num_factors,
        epochs,
        use_global_intercept,
        runName=run_name,
        logging=logging,
        flipFactorsForIdentification=flip_factors_for_identification,
        noteInit=note_init,
        userInit=user_init,
        user_variances=user_variances_for_mf,
        return_diagnostics=return_diagnostics,
    )

    if return_diagnostics:
        note_params, rater_params, global_bias_raw, epoch_diagnostics = result
    else:
        note_params, rater_params, global_bias_raw = result
        epoch_diagnostics = None

    global_bias = _extract_global_bias(global_bias_raw)
    metadata.update(
        {
            "l2_lambda": float(l2_lambda),
            "l2_intercept_multiplier": float(l2_intercept_multiplier),
            "num_factors": int(num_factors),
            "epochs": int(epochs),
            "use_global_intercept": bool(use_global_intercept),
            "variance_floor": float(variance_floor),
            "weight_percentile": weight_percentile,
            "weight_percentile_strategy": weight_percentile_strategy,
            "n_large_weights_above_threshold": int(
                user_variances_for_mf["largeWeightAboveThreshold"].sum()
            ),
            "n_large_weights_reverted_to_mean": int(
                user_variances_for_mf["largeWeightRevertedToMean"].sum()
            ),
            "n_large_weights_clipped_to_threshold": int(
                user_variances_for_mf["largeWeightClippedToThreshold"].sum()
            ),
            "large_weight_threshold": float(
                user_variances_for_mf["largeWeightThreshold"].dropna().iloc[0]
            )
            if user_variances_for_mf["largeWeightThreshold"].notna().any()
            else np.nan,
            "large_weight_replacement_mean": float(
                user_variances_for_mf["largeWeightReplacementMean"].dropna().iloc[0]
            )
            if user_variances_for_mf["largeWeightReplacementMean"].notna().any()
            else np.nan,
            "large_weight_replacement_value": float(
                user_variances_for_mf["largeWeightReplacementValue"].dropna().iloc[0]
            )
            if user_variances_for_mf["largeWeightReplacementValue"].notna().any()
            else np.nan,
            "weight_normalization_scale": float(
                user_variances_for_mf["weightNormalizationScale"].iloc[0]
            ),
            "rating_weighted_mean_weight": float(
                np.average(
                    user_variances_for_mf["inverseVarianceWeight"],
                    weights=user_variances_for_mf["nRatingsForWeightScale"],
                )
            ),
            "run_name": run_name,
        }
    )

    return WeightedMFResult(
        global_bias=global_bias,
        note_params=note_params,
        rater_params=rater_params,
        user_variances_for_mf=user_variances_for_mf,
        metadata=metadata,
        epoch_diagnostics=epoch_diagnostics,
    )
