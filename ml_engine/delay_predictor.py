from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from ml_engine.feature_engineering import (
    build_features, build_single_flight_features,
    get_training_data, FEATURE_COLUMNS,
)


_MODELS_DIR = Path(__file__).parent / "models"
_CLASSIFIER_PATH = _MODELS_DIR / "delay_classifier.pkl"
_REGRESSOR_PATH = _MODELS_DIR / "delay_regressor.pkl"
_MODEL_METADATA_PATH = _MODELS_DIR / "model_metadata.pkl"
_CALIBRATOR_PATH = _MODELS_DIR / "delay_calibrator.pkl"

_calibrator_cache: Optional[object] = None
_models_cache: Optional[Tuple[Optional[XGBClassifier], Optional[XGBRegressor]]] = None

from app_config import (
    DELAY_THRESHOLD_MIN,
    CORR_WINDOW_DAYS,
    CORR_MIN_SAMPLES,
    CORR_DEADBAND_PROB,
    CORR_DEADBAND_MIN,
    CORR_LEARNING_RATE,
    CORR_MAX_LOGIT_BIAS,
    CORR_MAX_MIN_BIAS,
    CORR_CACHE_TTL_S,
    WEEKDAY_WEIGHT_DEFAULT,
    WEEKDAY_SAMPLE_FOR_FULL_CONFIDENCE,
    RISK_HIGH_PROB,
    RISK_MEDIUM_PROB,
    RISK_MEDIUM_PROB_BY_DOW,
    WEATHER_BASELINE,
    PREDICT_BLEND_WEIGHT,
)

# Keep local aliases so existing call sites and internal references are unchanged.
_DELAY_THRESHOLD_MIN = DELAY_THRESHOLD_MIN
_CORR_WINDOW_DAYS = CORR_WINDOW_DAYS
_CORR_MIN_SAMPLES = CORR_MIN_SAMPLES
_CORR_DEADBAND_PROB = CORR_DEADBAND_PROB
_CORR_DEADBAND_MIN = CORR_DEADBAND_MIN
_CORR_LEARNING_RATE = CORR_LEARNING_RATE
_CORR_MAX_LOGIT_BIAS = CORR_MAX_LOGIT_BIAS
_CORR_MAX_MIN_BIAS = CORR_MAX_MIN_BIAS
_CORR_CACHE_TTL_S = CORR_CACHE_TTL_S
_WEEKDAY_SAMPLE_FOR_FULL_CONFIDENCE = WEEKDAY_SAMPLE_FOR_FULL_CONFIDENCE
_RISK_HIGH_PROB = RISK_HIGH_PROB
_RISK_MEDIUM_PROB = RISK_MEDIUM_PROB
_RISK_MEDIUM_PROB_BY_DOW = dict(RISK_MEDIUM_PROB_BY_DOW)
_RISK_FLAG_THRESHOLD = _RISK_MEDIUM_PROB
_WEATHER_BASELINE = WEATHER_BASELINE

_correction_cache: Dict[str, Dict[str, Any]] = {}


def _resolve_correction_db_path(db_path: Optional[Path] = None) -> Optional[Path]:
    if db_path is not None:
        return db_path
    try:
        from data.opensky_db import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH
    except Exception:
        return None


def _load_route_correction(route: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = _resolve_correction_db_path(db_path)
    if path is None:
        return {}
    now = time.time()
    cached = _correction_cache.get(route)
    if cached and now - cached.get("_ts", 0) < _CORR_CACHE_TTL_S:
        return cached
    row = {}
    try:
        import sqlite3
        conn = sqlite3.connect(str(path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            r = conn.execute(
                "SELECT logit_bias, minutes_bias FROM prediction_correction WHERE route = ?",
                (route,),
            ).fetchone()
            if r:
                row = {
                    "logit_bias": float(r["logit_bias"] or 0.0),
                    "minutes_bias": float(r["minutes_bias"] or 0.0),
                }
        finally:
            conn.close()
    except Exception:
        row = {}
    row["_ts"] = now
    _correction_cache[route] = row
    return row


def _invalidate_correction_cache() -> None:
    _correction_cache.clear()


def apply_online_correction(
    prob: float,
    expected_min: float,
    route: str,
    db_path: Optional[Path] = None,
) -> tuple:
    """Apply the per-route online bias to a raw (prob, expected_min) pair.

    prob  -> sigmoid(logit(prob) + logit_bias)
    minutes -> expected_min + minutes_bias (floor 0)
    """
    corr = _load_route_correction(route, db_path=db_path)
    logit_bias = float(corr.get("logit_bias", 0.0))
    minutes_bias = float(corr.get("minutes_bias", 0.0))
    if logit_bias == 0.0 and minutes_bias == 0.0:
        return prob, expected_min
    if logit_bias != 0.0:
        p = max(1e-6, min(1.0 - 1e-6, float(prob)))
        logit = np.log(p / (1.0 - p)) + logit_bias
        prob = 1.0 / (1.0 + np.exp(-logit))
    prob = min(0.95, max(0.05, prob))
    minutes = max(0.0, float(expected_min) + minutes_bias)
    return prob, minutes


def update_online_correction(
    db_path: Optional[Path] = None,
    window_days: int = _CORR_WINDOW_DAYS,
) -> Dict[str, Any]:
    """Recompute per-route correction biases from recent prediction-vs-actual drift.

    Reads prediction_log rows whose outcome has been recorded (actual_is_delayed
    NOT NULL) within the window, groups by origin_destination, and for each route
    with >= min samples and drift beyond the deadband moves the bias a fraction
    (learning rate) of the way toward the target. Nothing happens for routes
    that are within the deadband, so small noise does not cause constant drift.
    """
    import sqlite3

    path = _resolve_correction_db_path(db_path)
    if path is None or not Path(str(path)).exists():
        return {"status": "no_db", "routes_adjusted": 0}
    rows = []
    try:
        conn = sqlite3.connect(str(path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT origin, destination, delay_probability, expected_delay_min,
                          actual_is_delayed, actual_delay_min
                   FROM prediction_log
                   WHERE actual_is_delayed IS NOT NULL
                     AND date >= date('now', ?)""",
                (f"-{int(window_days)} days",),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return {"status": "error", "routes_adjusted": 0}

    grouped: Dict[str, List[tuple]] = {}
    for r in rows:
        origin = (r["origin"] or "").strip()
        destination = (r["destination"] or "").strip()
        if not origin or not destination:
            continue
        route = f"{origin}_{destination}"
        prob = r["delay_probability"]
        expected = r["expected_delay_min"]
        if prob is None:
            continue
        actual_delay = 1 if r["actual_is_delayed"] else 0
        actual_min = r["actual_delay_min"] if r["actual_delay_min"] is not None else 0.0
        grouped.setdefault(route, []).append(
            (float(prob), float(expected or 0.0), actual_delay, float(actual_min))
        )

    adjusted = []
    for route, items in grouped.items():
        n = len(items)
        if n < _CORR_MIN_SAMPLES:
            continue
        mean_prob = sum(x[0] for x in items) / n
        mean_exp = sum(x[1] for x in items) / n
        actual_rate = sum(x[2] for x in items) / n
        actual_mean_min = sum(x[3] for x in items) / n
        prob_drift = actual_rate - mean_prob
        min_drift = actual_mean_min - mean_exp

        new_logit_bias = 0.0
        new_min_bias = 0.0
        if abs(prob_drift) > _CORR_DEADBAND_PROB:
            target = min(0.95, max(0.05, actual_rate))
            target_logit = np.log(target / (1.0 - target))
            current_logit = np.log(max(1e-6, min(1.0 - 1e-6, mean_prob)) / (1.0 - max(1e-6, min(1.0 - 1e-6, mean_prob))))
            new_logit_bias = (_CORR_LEARNING_RATE * (target_logit - current_logit))
        if abs(min_drift) > _CORR_DEADBAND_MIN:
            new_min_bias = _CORR_LEARNING_RATE * min_drift

        if new_logit_bias == 0.0 and new_min_bias == 0.0:
            continue
        new_logit_bias = max(-_CORR_MAX_LOGIT_BIAS, min(_CORR_MAX_LOGIT_BIAS, new_logit_bias))
        new_min_bias = max(-_CORR_MAX_MIN_BIAS, min(_CORR_MAX_MIN_BIAS, new_min_bias))
        residual = prob_drift if abs(prob_drift) > abs(min_drift) / 100.0 else min_drift
        try:
            conn = sqlite3.connect(str(path), timeout=5)
            try:
                conn.execute(
                    """INSERT INTO prediction_correction
                       (route, logit_bias, minutes_bias, sample_count, last_residual, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(route) DO UPDATE SET
                         logit_bias = prediction_correction.logit_bias + excluded.logit_bias,
                         minutes_bias = prediction_correction.minutes_bias + excluded.minutes_bias,
                         sample_count = excluded.sample_count,
                         last_residual = excluded.last_residual,
                         updated_at = excluded.updated_at""",
                    (
                        route, round(new_logit_bias, 4), round(new_min_bias, 4),
                        n, round(prob_drift, 4),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            continue
        adjusted.append({
            "route": route, "samples": n, "prob_drift": round(prob_drift, 4),
            "min_drift": round(min_drift, 1),
            "logit_bias_delta": round(new_logit_bias, 4),
            "minutes_bias_delta": round(new_min_bias, 4),
        })

    _invalidate_correction_cache()
    return {
        "status": "success",
        "routes_with_actuals": len(grouped),
        "routes_adjusted": len(adjusted),
        "window_days": int(window_days),
        "adjustments": adjusted,
    }


def _reset_online_corrections(db_path: Optional[Path] = None) -> None:
    """Zero out all correction biases (called after a full model retrain)."""
    import sqlite3

    path = _resolve_correction_db_path(db_path)
    if path is None:
        return
    try:
        conn = sqlite3.connect(str(path), timeout=5)
        try:
            conn.execute("DELETE FROM prediction_correction")
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
    _invalidate_correction_cache()




def _compute_weather_adjustment(
    wind_speed_kmh: float, wind_gusts_kmh: float,
    visibility_m: float, precipitation_mm: float,
    cloud_cover_pct: float, departure_hour: int,
) -> Dict[str, float]:
    prob_boost = 0.0
    delay_boost = 0.0

    wind_ratio = max(0, (wind_speed_kmh - _WEATHER_BASELINE["wind_kmh"]) / _WEATHER_BASELINE["wind_kmh"])
    if wind_speed_kmh > 35:
        prob_boost += min(0.25, wind_ratio * 0.3)
        delay_boost += min(15, wind_ratio * 20)
    elif wind_speed_kmh > 25:
        prob_boost += min(0.10, wind_ratio * 0.15)
        delay_boost += min(5, wind_ratio * 8)

    gust_ratio = max(0, (wind_gusts_kmh - 50) / 50)
    if wind_gusts_kmh > 50:
        prob_boost += min(0.15, gust_ratio * 0.2)
        delay_boost += min(10, gust_ratio * 15)

    if precipitation_mm > 0.5:
        precip_severity = min(1.0, precipitation_mm / 10)
        prob_boost += precip_severity * 0.20
        delay_boost += precip_severity * 12
    if precipitation_mm > 5:
        prob_boost += 0.05
        delay_boost += 5

    if visibility_m < 5000:
        vis_severity = max(0, (5000 - visibility_m) / 5000)
        prob_boost += vis_severity * 0.15
        delay_boost += vis_severity * 10
    if visibility_m < 1000:
        prob_boost += 0.10
        delay_boost += 8

    if cloud_cover_pct > 95:
        prob_boost += 0.03
        delay_boost += 2
    elif cloud_cover_pct > 80:
        prob_boost += 0.01

    if departure_hour in (8, 9, 17, 18, 19, 20):
        prob_boost += 0.03
        delay_boost += 2

    if wind_speed_kmh > 40 and precipitation_mm > 5 and visibility_m < 2000:
        prob_boost += 0.10
        delay_boost += 8

    return {
        "probability_boost": round(prob_boost, 3),
        "delay_boost": round(delay_boost, 1),
    }


def _ensure_models_dir() -> None:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_models() -> Tuple[Optional[XGBClassifier], Optional[XGBRegressor]]:
    global _models_cache
    if _models_cache is not None:
        return _models_cache
    if not _CLASSIFIER_PATH.exists() or not _REGRESSOR_PATH.exists():
        return None, None
    try:
        clf = joblib.load(_CLASSIFIER_PATH)
        reg = joblib.load(_REGRESSOR_PATH)
        _models_cache = (clf, reg)
        return _models_cache
    except Exception:
        return None, None


def _save_models(clf: XGBClassifier, reg: XGBRegressor, metadata: Dict[str, Any]) -> None:
    global _models_cache, _calibrator_cache
    _ensure_models_dir()
    joblib.dump(clf, _CLASSIFIER_PATH)
    joblib.dump(reg, _REGRESSOR_PATH)
    joblib.dump(metadata, _MODEL_METADATA_PATH)
    _models_cache = (clf, reg)
    _calibrator_cache = None


def _save_metadata(metadata: Dict[str, Any]) -> None:
    _ensure_models_dir()
    joblib.dump(metadata, _MODEL_METADATA_PATH)


def _load_metadata() -> Dict[str, Any]:
    if _MODEL_METADATA_PATH.exists():
        try:
            return joblib.load(_MODEL_METADATA_PATH)
        except Exception:
            pass
    return {}


def get_prediction_threshold() -> float:
    """Single source of truth for the binary 'predicted delayed' threshold.

    Returns the F1-optimal threshold fit on calibrated out-of-fold blended
    scores at training time (stored in model metadata). Falls back to 0.30
    when no model exists yet.
    """
    meta = _load_metadata()
    ot = meta.get("optimal_threshold")
    if isinstance(ot, dict):
        t = ot.get("threshold")
        if isinstance(t, (int, float)) and 0.0 <= t <= 1.0:
            return float(t)
    return 0.30


def _load_calibrator():
    global _calibrator_cache
    if _calibrator_cache is not None:
        return _calibrator_cache
    if not _CALIBRATOR_PATH.exists():
        return None
    try:
        _calibrator_cache = joblib.load(_CALIBRATOR_PATH)
        return _calibrator_cache
    except Exception:
        return None


def _compute_class_weights(y: pd.Series) -> Dict[int, float]:
    counts = y.value_counts()
    total = len(y)
    weights = {}
    for cls in counts.index:
        weights[int(cls)] = total / (len(counts) * counts[cls])
    return weights


def _best_delay_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    thresholds = np.arange(0.05, 0.901, 0.01)
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"] or (f1 == best["f1"] and t < best["threshold"]):
            best = {
                "threshold": round(float(t), 3),
                "f1": round(float(f1), 4),
                "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            }
    return best


def _register_model_run(metadata: Dict[str, Any], db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Snapshot the active artifacts under a version id and log the run.

    Called after a successful training run so the model can be rolled back via
    data.opensky_db.activate_model_version. Failures here never block training.
    """
    try:
        from data.opensky_db import register_model_run
    except Exception:
        return {"status": "skipped"}
    version = "v" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_files: Dict[str, str] = {}
    for fname in ("delay_classifier.pkl", "delay_regressor.pkl", "delay_calibrator.pkl", "model_metadata.pkl"):
        src = _MODELS_DIR / fname
        if not src.exists():
            continue
        snap = f"{Path(fname).stem}_{version}.pkl"
        try:
            import shutil
            shutil.copy2(str(src), str(_MODELS_DIR / snap))
            artifact_files[fname] = snap
        except Exception:
            pass
    try:
        return register_model_run(version, metadata, artifact_files, db_path=db_path)
    except Exception:
        return {"status": "error"}


def train_model(
    min_samples: int = 30,
    callsigns: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if callsigns:
        min_samples = min(min_samples, 5)
    features, binary_target, reg_target, route_labels, dates = get_training_data(
        min_samples=min_samples, callsigns=callsigns, db_path=db_path,
        with_dates=True,
    )

    if len(features) < min_samples:
        return {
            "status": "insufficient_data",
            "samples": len(features),
            "min_required": min_samples,
            "message": f"Need {min_samples} samples, have {len(features)}. Using heuristic fallback.",
        }

    from sklearn.model_selection import GroupKFold, train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_absolute_error, mean_squared_error, r2_score,
        confusion_matrix, roc_auc_score, log_loss,
        average_precision_score, matthews_corrcoef,
        cohen_kappa_score, balanced_accuracy_score,
    )

    class_weights = _compute_class_weights(binary_target)
    sample_weights = binary_target.map(class_weights).values

    delayed_mask = (binary_target == 1).to_numpy()
    reg_features = features[delayed_mask]
    reg_target_delayed = reg_target[delayed_mask].clip(lower=0, upper=240)
    reg_groups = route_labels[delayed_mask].reset_index(drop=True)

    n_splits = min(5, route_labels.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    clf_metrics_list = []
    oof_y: List[int] = []
    oof_proba: List[float] = []

    for train_idx, test_idx in gkf.split(features, binary_target, groups=route_labels):
        X_tr, X_te = features.iloc[train_idx], features.iloc[test_idx]
        y_tr_clf, y_te_clf = binary_target.iloc[train_idx], binary_target.iloc[test_idx]
        sw_tr = sample_weights[train_idx]

        pos_weight = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)

        clf = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            eval_metric="aucpr", random_state=42,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            min_child_weight=2,
            subsample=0.9, colsample_bytree=0.9,
        )
        clf.fit(X_tr, y_tr_clf, sample_weight=sw_tr)

        y_te_pred = clf.predict(X_te)
        y_te_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else y_te_pred.astype(float)

        oof_y.extend(y_te_clf.astype(int).tolist())
        oof_proba.extend(y_te_proba.astype(float).tolist())

        n_delayed_te = int(y_te_clf.sum())
        n_clean_te = int(len(y_te_clf) - y_te_clf.sum())
        has_both_classes = n_delayed_te > 0 and n_clean_te > 0
        clf_m = {
            "accuracy": accuracy_score(y_te_clf, y_te_pred),
            "balanced_accuracy": balanced_accuracy_score(y_te_clf, y_te_pred),
            "precision": precision_score(y_te_clf, y_te_pred, zero_division=0),
            "recall": recall_score(y_te_clf, y_te_pred, zero_division=0),
            "f1": f1_score(y_te_clf, y_te_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_te_clf, y_te_pred),
            "kappa": cohen_kappa_score(y_te_clf, y_te_pred),
            "roc_auc": roc_auc_score(y_te_clf, y_te_proba) if has_both_classes else None,
            "pr_auc": average_precision_score(y_te_clf, y_te_proba) if has_both_classes else None,
            "log_loss": log_loss(y_te_clf, y_te_proba, labels=[0, 1]) if has_both_classes else None,
            "confusion": confusion_matrix(y_te_clf, y_te_pred).tolist(),
            "samples": len(y_te_clf),
            "delayed_in_test": n_delayed_te,
        }
        clf_metrics_list.append(clf_m)

    reg_metrics_list = []
    if len(reg_features) >= 20 and reg_groups.nunique() >= 2:
        n_splits_reg = min(5, reg_groups.nunique())
        gkf_reg = GroupKFold(n_splits=n_splits_reg)
        for tr, te in gkf_reg.split(reg_features, reg_target_delayed, groups=reg_groups):
            reg = XGBRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                random_state=42, reg_alpha=0.1, reg_lambda=1.0,
                min_child_weight=2,
                subsample=0.9, colsample_bytree=0.9,
            )
            reg.fit(reg_features.iloc[tr], reg_target_delayed.iloc[tr])
            p = reg.predict(reg_features.iloc[te])
            reg_metrics_list.append({
                "mae": mean_absolute_error(reg_target_delayed.iloc[te], p),
                "rmse": float(np.sqrt(mean_squared_error(reg_target_delayed.iloc[te], p))),
                "r2": r2_score(reg_target_delayed.iloc[te], p),
            })

    def _avg_dicts(dicts, keys=None):
        if not dicts:
            return {}
        if keys is None:
            keys = []
            for d in dicts:
                for k, v in d.items():
                    if isinstance(v, (int, float)) and v is not None and k not in keys:
                        keys.append(k)
        result = {}
        for k in keys:
            vals = [d[k] for d in dicts if k in d and d[k] is not None]
            result[k] = round(np.mean(vals), 4) if vals else None
        return result

    avg_clf = _avg_dicts(clf_metrics_list)
    avg_reg = _avg_dicts(reg_metrics_list)

    from sklearn.isotonic import IsotonicRegression

    oof_y_arr = np.asarray(oof_y, dtype=float)
    oof_proba_arr = np.asarray(oof_proba, dtype=float)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(oof_proba_arr, oof_y_arr)
    cal_oof_proba = calibrator.predict(oof_proba_arr)
    oof_threshold = _best_delay_threshold(oof_y_arr, cal_oof_proba)
    joblib.dump(calibrator, _CALIBRATOR_PATH)

    overall_rate = float(binary_target.mean())
    prior_oof = features.apply(
        lambda row: _prior_probability(
            dow_delay_rate=row["dow_delay_rate"],
            dow_sample_count=row["dow_delay_count"],
            route_delay_rate=row["route_delay_rate"],
            route_hour_delay_rate=row["route_hour_delay_rate"],
            callsign_delay_rate=row["callsign_delay_rate"],
            overall_delay_rate=overall_rate,
        ),
        axis=1,
    ).to_numpy()

    deployed_oof = np.clip(0.7 * prior_oof + 0.3 * cal_oof_proba, 0.05, 0.95)
    oof_threshold = _best_delay_threshold(oof_y_arr, deployed_oof)
    blend_weight_selected = 0.7

    temporal_metrics = {}
    if dates.notna().sum() > 0:
        try:
            unique_dates = sorted(dates.dropna().dt.normalize().unique())
            if len(unique_dates) >= 3:
                n_test_days = max(1, int(round(len(unique_dates) * 0.3)))
                cutoff = unique_dates[-n_test_days]
                te_mask = (dates.dt.normalize() >= cutoff).to_numpy()
                tr_mask = ~te_mask
                y_te_t = binary_target.to_numpy()[te_mask].astype(int)
                y_tr_t = binary_target.to_numpy()[tr_mask].astype(int)
                if te_mask.sum() >= 10 and tr_mask.sum() >= 10 and y_te_t.sum() > 0 and y_tr_t.sum() > 0:
                    clf_temporal = XGBClassifier(
                        n_estimators=150, max_depth=6, learning_rate=0.1,
                        eval_metric="aucpr", random_state=42,
                        reg_alpha=0.1, reg_lambda=1.0,
                        scale_pos_weight=class_weights.get(1, 1.0) / class_weights.get(0, 1.0),
                        min_child_weight=2,
                        subsample=0.9, colsample_bytree=0.9,
                    )
                    clf_temporal.fit(features.iloc[tr_mask], binary_target.iloc[tr_mask],
                                     sample_weight=sample_weights[tr_mask])
                    proba_te_t = clf_temporal.predict_proba(features.iloc[te_mask])[:, 1]
                    cal_te_t = calibrator.predict(proba_te_t)
                    th_t = oof_threshold["threshold"]
                    pred_te_t = (cal_te_t >= th_t).astype(int)
                    temporal_metrics = {
                        "test_start": str(unique_dates[-n_test_days]),
                        "test_end": str(unique_dates[-1]),
                        "train_end": str(unique_dates[-n_test_days - 1]),
                        "test_samples": int(len(y_te_t)),
                        "train_samples": int(len(y_tr_t)),
                        "test_delay_rate": round(float(y_te_t.mean()), 4),
                        "threshold": round(float(th_t), 4),
                        "accuracy": round(float(accuracy_score(y_te_t, pred_te_t)), 4),
                        "precision": round(float(precision_score(y_te_t, pred_te_t, zero_division=0)), 4),
                        "recall": round(float(recall_score(y_te_t, pred_te_t, zero_division=0)), 4),
                        "f1": round(float(f1_score(y_te_t, pred_te_t, zero_division=0)), 4),
                        "roc_auc": round(float(roc_auc_score(y_te_t, cal_te_t)), 4),
                        "pr_auc": round(float(average_precision_score(y_te_t, cal_te_t)), 4),
                    }
        except Exception as exc:
            temporal_metrics = {"error": str(exc)}

    fold_clf_accs = [d["accuracy"] for d in clf_metrics_list]
    fold_clf_f1s = [d["f1"] for d in clf_metrics_list]
    fold_clf_aucs = [d["roc_auc"] for d in clf_metrics_list if d["roc_auc"] is not None]

    train_idx_final, test_idx_final = next(gkf.split(features, binary_target, groups=route_labels))
    X_train_final = features.iloc[train_idx_final]
    X_test_final = features.iloc[test_idx_final]
    y_train_final = binary_target.iloc[train_idx_final]
    y_test_final = binary_target.iloc[test_idx_final]
    sw_train_final = sample_weights[train_idx_final]

    pos_weight_final = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)

    clf_final = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        eval_metric="aucpr", random_state=42,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=pos_weight_final,
        min_child_weight=2,
        subsample=0.9, colsample_bytree=0.9,
    )
    clf_final.fit(X_train_final, y_train_final, sample_weight=sw_train_final)

    if len(reg_features) >= 20:
        reg_final = XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            random_state=42, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=2,
            subsample=0.9, colsample_bytree=0.9,
        )
        reg_final.fit(reg_features, reg_target_delayed)
    else:
        reg_final = XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            random_state=42, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=2,
            subsample=0.9, colsample_bytree=0.9,
        )
        reg_final.fit(features, reg_target.clip(lower=0, upper=240))

    feature_importance = dict(zip(FEATURE_COLUMNS, clf_final.feature_importances_.tolist()))

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "samples": len(features),
        "train_samples": len(X_train_final),
        "test_samples": len(X_test_final),
        "class_weights": {str(k): round(v, 4) for k, v in class_weights.items()},
        "delay_rate": round(float(binary_target.mean()), 4),
        "n_routes": int(route_labels.nunique()),
        "n_folds": n_splits,
        "classifier_cv": {k: round(v, 4) if isinstance(v, float) else v for k, v in avg_clf.items() if k != "confusion"},
        "classifier_test": {k: round(v, 4) if isinstance(v, float) else v for k, v in avg_clf.items() if k != "confusion"},
        "regressor_cv": {k: round(v, 4) for k, v in avg_reg.items()},
        "optimal_threshold": oof_threshold,
        "blend_weight": blend_weight_selected,
        "temporal_test": temporal_metrics,
        "fold_accuracy": [round(a, 4) for a in fold_clf_accs],
        "fold_f1": [round(f, 4) for f in fold_clf_f1s],
        "fold_roc_auc": [round(a, 4) for a in fold_clf_aucs],
        "feature_importance": {k: round(v, 4) for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)},
        "feature_columns": FEATURE_COLUMNS,
    }

    _save_models(clf_final, reg_final, metadata)
    _reset_online_corrections(db_path)
    _register_model_run(metadata, db_path)

    cv_acc = avg_clf.get("accuracy", 0)
    cv_f1 = avg_clf.get("f1", 0)
    cv_auc = avg_clf.get("roc_auc")
    cv_mcc = avg_clf.get("mcc", 0)
    cv_mae = avg_reg.get("mae", 0)

    auc_str = f"{cv_auc:.4f}" if cv_auc is not None else "N/A"
    return {
        "status": "success",
        "samples": len(features),
        "train_samples": len(X_train_final),
        "test_samples": len(X_test_final),
        "n_routes": int(route_labels.nunique()),
        "delay_rate": round(float(binary_target.mean()), 4),
        "class_weights": {str(k): round(v, 4) for k, v in class_weights.items()},
        "cv_accuracy": round(cv_acc, 4),
        "cv_balanced_accuracy": round(avg_clf.get("balanced_accuracy", 0), 4),
        "cv_precision": round(avg_clf.get("precision", 0), 4),
        "cv_recall": round(avg_clf.get("recall", 0), 4),
        "cv_f1": round(cv_f1, 4),
        "cv_mcc": round(cv_mcc, 4),
        "cv_kappa": round(avg_clf.get("kappa", 0), 4),
        "cv_roc_auc": round(cv_auc, 4) if cv_auc is not None else None,
        "cv_pr_auc": round(avg_clf.get("pr_auc", 0), 4) if avg_clf.get("pr_auc") is not None else None,
        "cv_log_loss": round(avg_clf.get("log_loss", 0), 4) if avg_clf.get("log_loss") is not None else None,
        "optimal_threshold": oof_threshold["threshold"],
        "threshold_f1": oof_threshold["f1"],
        "threshold_precision": oof_threshold["precision"],
        "threshold_recall": oof_threshold["recall"],
        "temporal_test": temporal_metrics,
        "fold_accuracy": [round(a, 4) for a in fold_clf_accs],
        "fold_f1": [round(f, 4) for f in fold_clf_f1s],
        "fold_roc_auc": [round(a, 4) for a in fold_clf_aucs],
        "cv_mae": round(cv_mae, 2),
        "cv_rmse": round(avg_reg.get("rmse", 0), 2),
        "cv_r2": round(avg_reg.get("r2", 0), 4),
        "feature_importance": metadata["feature_importance"],
        "message": (
            f"Trained on {len(features)} samples ({n_splits}-fold GroupKFold by route). "
            f"CV Accuracy: {cv_acc:.1%}, F1: {cv_f1:.4f}, MCC: {cv_mcc:.4f}, "
            f"ROC-AUC: {auc_str}, Regression MAE: {cv_mae:.1f}min"
        ),
    }


def retrain_if_stale(max_age_hours: int = 24, callsigns: Optional[List[str]] = None, db_path: Optional[Path] = None) -> Dict[str, Any]:
    metadata = _load_metadata()
    if not metadata:
        return train_model(callsigns=callsigns, db_path=db_path)

    trained_at = metadata.get("trained_at")
    if trained_at:
        try:
            dt = datetime.fromisoformat(trained_at)
            if datetime.now() - dt < timedelta(hours=max_age_hours):
                return {"status": "up_to_date", "message": "Model is current."}
        except (ValueError, TypeError):
            pass

    return train_model(callsigns=callsigns, db_path=db_path)


def _predict_with_ml(features_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    clf, reg = _load_models()
    if clf is None or reg is None:
        return None

    try:
        prob = clf.predict_proba(features_df)[0]
        raw_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
        cal = _load_calibrator()
        if cal is not None:
            delay_prob = float(cal.predict(np.asarray([raw_prob]))[0])
        else:
            delay_prob = raw_prob
        delay_prob = max(0.0, min(1.0, delay_prob))
        cond_delay = float(reg.predict(features_df)[0])
        cond_delay = max(0.0, min(240.0, cond_delay))
        expected_delay = round(delay_prob * cond_delay, 1)

        feature_importance = {}
        if hasattr(clf, "feature_importances_"):
            for i, col in enumerate(FEATURE_COLUMNS):
                if i < len(clf.feature_importances_):
                    feature_importance[col] = round(float(clf.feature_importances_[i]), 4)

        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "delay_probability": round(delay_prob, 3),
            "expected_delay_min": round(expected_delay, 1),
            "model_used": "xgboost",
            "top_influencing_features": [{"feature": k, "importance": v} for k, v in top_features],
        }
    except Exception:
        return None


def _prior_probability(
    dow_delay_rate: float,
    dow_sample_count: int,
    route_delay_rate: float,
    route_hour_delay_rate: float,
    callsign_delay_rate: float,
    overall_delay_rate: float = 0.0,
) -> float:
    base_rate = 0.075 if overall_delay_rate <= 0 else overall_delay_rate
    conf = min(1.0, (dow_sample_count or 0) / 3.0)
    if dow_delay_rate is None:
        dow_delay_rate = 0.0
    dow_term = dow_delay_rate * conf + base_rate * (1 - conf)
    route_term = route_delay_rate if route_delay_rate else base_rate
    hour_term = route_hour_delay_rate if route_hour_delay_rate else route_term
    callsign_term = callsign_delay_rate if callsign_delay_rate else route_term
    prob = 0.45 * dow_term + 0.25 * route_term + 0.15 * callsign_term + 0.15 * hour_term
    return max(0.02, min(0.95, prob))


def _heuristic_predict(
    origin: str,
    destination: str,
    aircraft_type: str,
    departure_hour: int,
    pax_count: int,
    flight_duration_min: int,
    is_international: bool,
) -> Dict[str, Any]:
    hour_risk_map = {
        0: 0.40, 1: 0.42, 2: 0.45, 3: 0.48, 4: 0.30, 5: 0.15,
        6: 0.12, 7: 0.18, 8: 0.25, 9: 0.22, 10: 0.20, 11: 0.22,
        12: 0.28, 13: 0.25, 14: 0.22, 15: 0.20, 16: 0.25, 17: 0.30,
        18: 0.35, 19: 0.38, 20: 0.42, 21: 0.45, 22: 0.48, 23: 0.44,
    }

    hour_risk = hour_risk_map.get(departure_hour, 0.25)
    pax_risk = min(pax_count / 200.0, 1.0) * 0.06
    duration_risk = min(flight_duration_min / 300.0, 1.0) * 0.08
    intl_risk = 0.04 if is_international else 0.0
    peak_risk = 0.10 if departure_hour in (8, 9, 17, 18, 19, 20) else 0.0

    prob = min(0.95, max(0.05, hour_risk + pax_risk + duration_risk + intl_risk + peak_risk))
    expected_delay = 0.0
    if prob > 0.3:
        expected_delay = prob * 35.0
    elif prob > 0.15:
        expected_delay = prob * 20.0

    return {
        "delay_probability": round(prob, 3),
        "expected_delay_min": round(expected_delay, 1),
        "model_used": "heuristic",
        "top_influencing_features": [],
    }


def predict_delay(
    origin: str = "BLR",
    destination: str = "DEL",
    aircraft_type: str = "B737",
    departure_hour: int = 12,
    pax_count: int = 150,
    flight_duration_min: int = 120,
    is_international: bool = False,
    departure_time: Optional[datetime] = None,
    turbulence_category: str = "",
    seat_occupancy: float = 0.5,
    distance: float = 2000.0,
    prev_flight_delay: float = 0.0,
    wind_speed_kmh: float = 0.0,
    wind_gusts_kmh: float = 0.0,
    visibility_m: float = 10000.0,
    cloud_cover_pct: float = 0.0,
    precipitation_mm: float = 0.0,
    temperature_c: float = 25.0,
    pressure_hpa: float = 1013.0,
    delay_rate_pct: Optional[float] = None,
    dow_delay_rate: Optional[float] = None,
    overall_delay_rate: Optional[float] = None,
    weekday_weight: float = WEEKDAY_WEIGHT_DEFAULT,
    dow_sample_count: Optional[int] = None,
    expected_duration_min: float = 0.0,
    route_hour_delay_rate: float = 0.0,
    route_delay_rate: float = 0.0,
    callsign_delay_rate: float = 0.0,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if departure_time is None:
        departure_time = datetime.now()

    route = f"{origin}_{destination}"

    wind_speed_kmh = float(wind_speed_kmh or 0)
    wind_gusts_kmh = float(wind_gusts_kmh or 0)
    visibility_m = float(visibility_m or 10000)
    cloud_cover_pct = float(cloud_cover_pct or 0)
    precipitation_mm = float(precipitation_mm or 0)
    temperature_c = float(temperature_c or 25)
    pressure_hpa = float(pressure_hpa or 1013)

    features_df = build_single_flight_features(
        hour_of_day=departure_hour,
        day_of_week=departure_time.weekday(),
        month=departure_time.month,
        route=route,
        prev_delay=prev_flight_delay,
        wind_speed_kmh=wind_speed_kmh,
        wind_gusts_kmh=wind_gusts_kmh,
        visibility_m=visibility_m,
        cloud_cover_pct=cloud_cover_pct,
        precipitation_mm=precipitation_mm,
        temperature_c=temperature_c,
        pressure_hpa=pressure_hpa,
        expected_duration_min=expected_duration_min,
        route_hour_delay_rate=route_hour_delay_rate,
        route_delay_rate=route_delay_rate,
        callsign_delay_rate=callsign_delay_rate,
        dow_delay_rate=dow_delay_rate or 0,
        dow_delay_count=dow_sample_count or 0,
    )

    ml_result = _predict_with_ml(features_df)
    if ml_result:
        result = ml_result
    else:
        result = _heuristic_predict(
            origin, destination, aircraft_type, departure_hour,
            pax_count, flight_duration_min, is_international,
        )

    prior_prob = _prior_probability(
        dow_delay_rate=dow_delay_rate if dow_delay_rate is not None else 0,
        dow_sample_count=dow_sample_count or 0,
        route_delay_rate=route_delay_rate,
        route_hour_delay_rate=route_hour_delay_rate,
        callsign_delay_rate=callsign_delay_rate,
        overall_delay_rate=overall_delay_rate or 0,
    )

    meta = _load_metadata()
    blend_w = PREDICT_BLEND_WEIGHT
    if meta:
        bw = meta.get("blend_weight")
        if isinstance(bw, (int, float)) and 0 <= bw <= 1:
            blend_w = float(bw)

    old_prob = result["delay_probability"]
    if result["model_used"] == "xgboost":
        blended = blend_w * prior_prob + (1.0 - blend_w) * old_prob
    else:
        blended = prior_prob
    cond_magnitude = result["expected_delay_min"] / old_prob if old_prob > 0 else 0

    weather_adj = _compute_weather_adjustment(
        wind_speed_kmh=wind_speed_kmh, wind_gusts_kmh=wind_gusts_kmh,
        visibility_m=visibility_m, precipitation_mm=precipitation_mm,
        cloud_cover_pct=cloud_cover_pct, departure_hour=departure_hour,
    )
    adjusted_prob = max(0.05, min(0.95, blended + weather_adj["probability_boost"]))
    result["delay_probability"] = round(adjusted_prob, 3)
    result["expected_delay_min"] = round(adjusted_prob * cond_magnitude + weather_adj["delay_boost"], 1)

    corrected_prob, corrected_min = apply_online_correction(
        result["delay_probability"], result["expected_delay_min"], route, db_path=db_path
    )
    result["delay_probability"] = round(corrected_prob, 3)
    result["expected_delay_min"] = round(corrected_min, 1)

    prob = result["delay_probability"]
    exp_delay = result["expected_delay_min"]

    threshold = get_prediction_threshold()

    risk_score = prob * exp_delay
    if departure_time is not None:
        medium_threshold = _RISK_MEDIUM_PROB_BY_DOW.get(departure_time.weekday(), _RISK_MEDIUM_PROB)
    else:
        medium_threshold = _RISK_MEDIUM_PROB
    if prob >= _RISK_HIGH_PROB:
        risk_level = "High"
    elif prob >= medium_threshold:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    factors = []
    if departure_hour in (8, 9, 17, 18, 19, 20):
        factors.append("Peak hour departure increases delay risk")
    if departure_hour >= 22 or departure_hour < 5:
        factors.append("Night operations may face curfew/constraints")
    if pax_count > 160:
        factors.append("High passenger load extends boarding time")
    if is_international:
        factors.append("International flights require additional clearance")
    if prev_flight_delay > 15:
        factors.append(f"Aircraft arrived {prev_flight_delay:.0f} min late from previous flight")
    if wind_speed_kmh > 40:
        factors.append(f"High wind speed ({wind_speed_kmh:.0f} km/h) may cause delays")
    if wind_gusts_kmh > 60:
        factors.append(f"Strong wind gusts ({wind_gusts_kmh:.0f} km/h)")
    if visibility_m < 1000:
        factors.append(f"Low visibility ({visibility_m:.0f}m) may cause approach delays")
    if precipitation_mm > 5:
        factors.append(f"Heavy precipitation ({precipitation_mm:.1f}mm)")
    if cloud_cover_pct > 80:
        factors.append("Overcast conditions")
    if result["model_used"] == "xgboost":
        factors.append("Prediction based on trained ML model")
    else:
        factors.append("Prediction based on heuristic (model not yet trained)")

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if dow_delay_rate is not None and overall_delay_rate is not None and overall_delay_rate > 0:
        dow_pct = dow_delay_rate * 100
        overall_pct = overall_delay_rate * 100
        has_weekday_data = dow_sample_count is None or dow_sample_count > 0
        if departure_time is not None:
            day_name = day_names[departure_time.weekday()]
        else:
            day_name = "this day"
        if dow_pct > overall_pct * 1.2:
            factors.append(
                f"Recent-week history: {dow_pct:.0f}% delay rate on {day_name}s (previous weeks) vs {overall_pct:.0f}% overall — weekday pattern weighted in"
            )
        elif dow_pct < overall_pct * 0.8 and dow_pct > 0:
            factors.append(
                f"Lower recent-week delay rate on {day_name}s ({dow_pct:.0f}% vs {overall_pct:.0f}% overall) — weekday pattern weighted in"
            )
        elif dow_pct == 0 and overall_pct > 5 and has_weekday_data:
            factors.append(
                f"No recent delays on {day_name}s for this route (vs {overall_pct:.0f}% overall)"
            )

    result.update({
        "risk_level": risk_level,
        "predicted_delayed": bool(prob >= threshold),
        "factors": factors,
        "features": {
            "wind_speed_knots": round(wind_speed_kmh * 0.539957, 1),
            "wind_gust_knots": round(wind_gusts_kmh * 0.539957, 1),
            "visibility_m": visibility_m,
            "cloud_cover_pct": cloud_cover_pct,
            "precipitation_mm": precipitation_mm,
            "temperature_c": temperature_c,
            "pressure_hpa": pressure_hpa,
            "prev_flight_delay": prev_flight_delay,
        },
    })

    return result


def train_delay_model(flight_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(flight_data) < 5:
        return {
            "status": "insufficient_data",
            "message": "Need at least 5 flights to train. Using heuristic model.",
            "samples": len(flight_data),
        }

    correct = 0
    total = len(flight_data)
    for record in flight_data:
        predicted = predict_delay(
            origin=record.get("origin", "DEL"),
            destination=record.get("destination", "BOM"),
            aircraft_type=record.get("aircraft_type", "B737"),
            departure_hour=record.get("departure_hour", 12),
            pax_count=record.get("pax_count", 150),
            flight_duration_min=record.get("flight_duration_min", 120),
            is_international=record.get("is_international", False),
        )
        was_delayed = record.get("actual_delay_min", 0) > _DELAY_THRESHOLD_MIN
        predicted_delayed = predicted["predicted_delayed"]
        if was_delayed == predicted_delayed:
            correct += 1

    accuracy = correct / total
    return {
        "status": "trained",
        "accuracy": round(accuracy, 3),
        "samples": total,
        "correct": correct,
        "message": f"Model trained on {total} samples with {accuracy:.1%} accuracy.",
    }


def get_delay_insights(db_path: Optional[Path] = None) -> Dict[str, Any]:
    from data.opensky_db import get_flight_stats, get_feature_table
    stats = get_flight_stats(db_path)
    if stats.get("total_flights", 0) == 0:
        return {
            "total_flights": 0,
            "delayed_flights": 0,
            "delay_rate": 0,
            "message": "No OpenSky flight data available. Run 'Seed Historical Data' first.",
        }

    df = get_feature_table(db_path)
    if df.empty:
        return {
            "total_flights": stats["total_flights"],
            "delayed_flights": 0,
            "delay_rate": 0,
            "message": "Flight data exists but no delay labels computed.",
        }

    total = len(df)
    delayed = int(df["is_delayed"].sum()) if "is_delayed" in df.columns else 0
    delay_rate = delayed / total if total > 0 else 0

    by_hour = {}
    if "departure_hour" in df.columns:
        hour_counts = df[df["is_delayed"] == 1]["departure_hour"].value_counts()
        by_hour = {int(h): int(c) for h, c in hour_counts.items()}

    by_route = {}
    if "origin_airport" in df.columns and "destination_airport" in df.columns:
        route_delays = df[df["is_delayed"] == 1].groupby(
            ["origin_airport", "destination_airport"]
        ).size()
        for (o, d), c in route_delays.items():
            by_route[f"{o}-{d}"] = int(c)

    model_meta = _load_metadata()
    cv = model_meta.get("classifier_cv", {})

    return {
        "total_flights": total,
        "delayed_flights": delayed,
        "delay_rate": round(delay_rate, 3),
        "peak_delay_hours": [{"hour": h, "count": c} for h, c in sorted(by_hour.items(), key=lambda x: x[1], reverse=True)[:5]],
        "delay_by_route": by_route,
        "model_status": {
            "trained": model_meta.get("trained_at") is not None,
            "cv_accuracy": cv.get("accuracy"),
            "cv_f1": cv.get("f1"),
            "cv_mcc": cv.get("mcc"),
            "cv_roc_auc": cv.get("roc_auc"),
            "cv_mae": model_meta.get("regressor_cv", {}).get("mae"),
            "optimal_threshold": model_meta.get("optimal_threshold", {}).get("threshold"),
        },
    }


def get_delay_cause_breakdown(db_path: Optional[Path] = None) -> Dict[str, Any]:
    from data.opensky_db import get_feature_table
    df = get_feature_table(db_path)
    if df.empty:
        return {"message": "No data available."}

    causes = {}
    if "wind_speed_kmh" in df.columns:
        windy = df[df["wind_speed_kmh"] > 40]
        if len(windy) > 0:
            causes["high_wind"] = {"count": len(windy), "description": "Wind speed > 40 km/h"}

    if "precipitation_mm" in df.columns:
        rainy = df[df["precipitation_mm"] > 5]
        if len(rainy) > 0:
            causes["heavy_precipitation"] = {"count": len(rainy), "description": "Precipitation > 5mm"}

    if "visibility_m" in df.columns:
        low_vis = df[df["visibility_m"] < 1000]
        if len(low_vis) > 0:
            causes["low_visibility"] = {"count": len(low_vis), "description": "Visibility < 1000m"}

    if "prev_flight_delay_min" in df.columns:
        cascading = df[df["prev_flight_delay_min"] > 15]
        if len(cascading) > 0:
            causes["cascading_delay"] = {"count": len(cascading), "description": "Previous flight delay > 15 min"}

    return {
        "total_flights": len(df),
        "delayed_flights": int(df["is_delayed"].sum()) if "is_delayed" in df.columns else 0,
        "causes": causes,
    }


def get_delay_by_airport(db_path: Optional[Path] = None) -> Dict[str, Any]:
    from data.opensky_db import get_feature_table
    df = get_feature_table(db_path)
    if df.empty:
        return {"message": "No data available."}

    result = {}
    for airport in df["origin_airport"].dropna().unique():
        subset = df[df["origin_airport"] == airport]
        total = len(subset)
        delayed = int(subset["is_delayed"].sum()) if "is_delayed" in subset.columns else 0
        avg_dev = float(subset["deviation_min"].mean()) if "deviation_min" in subset.columns else 0
        result[airport] = {
            "total_flights": total,
            "delayed_flights": delayed,
            "delay_rate": round(delayed / total, 3) if total > 0 else 0,
            "avg_deviation_min": round(avg_dev, 1),
        }
    return result


def get_delay_by_time(db_path: Optional[Path] = None) -> Dict[str, Any]:
    from data.opensky_db import get_feature_table
    df = get_feature_table(db_path)
    if df.empty:
        return {"message": "No data available."}

    by_hour = {}
    if "departure_hour" in df.columns:
        for hour in range(24):
            subset = df[df["departure_hour"] == hour]
            if len(subset) > 0:
                delayed = int(subset["is_delayed"].sum()) if "is_delayed" in subset.columns else 0
                by_hour[hour] = {
                    "total": len(subset),
                    "delayed": delayed,
                    "rate": round(delayed / len(subset), 3),
                }

    by_dow = {}
    if "day_of_week" in df.columns:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for dow in range(7):
            subset = df[df["day_of_week"] == dow]
            if len(subset) > 0:
                delayed = int(subset["is_delayed"].sum()) if "is_delayed" in subset.columns else 0
                by_dow[day_names[dow]] = {
                    "total": len(subset),
                    "delayed": delayed,
                    "rate": round(delayed / len(subset), 3),
                }

    return {"by_hour": by_hour, "by_day_of_week": by_dow}


def get_delay_by_route_type(db_path: Optional[Path] = None) -> Dict[str, Any]:
    from data.opensky_db import get_feature_table
    df = get_feature_table(db_path)
    if df.empty:
        return {"message": "No data available."}

    by_route = {}
    if "origin_airport" in df.columns and "destination_airport" in df.columns:
        for route in df.groupby(["origin_airport", "destination_airport"]).groups:
            o, d = route
            subset = df[(df["origin_airport"] == o) & (df["destination_airport"] == d)]
            total = len(subset)
            delayed = int(subset["is_delayed"].sum()) if "is_delayed" in subset.columns else 0
            by_route[f"{o}-{d}"] = {
                "total": total,
                "delayed": delayed,
                "rate": round(delayed / total, 3) if total > 0 else 0,
            }

    return {"by_route_type": by_route}


def invalidate_profiles_cache() -> None:
    pass
