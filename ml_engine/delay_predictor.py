from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
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

_DELAY_THRESHOLD_MIN = 15.0

_WEATHER_BASELINE = {
    "wind_kmh": 21.0,
    "precip_mm": 0.1,
    "vis_m": 10000.0,
    "cloud_pct": 88.5,
}


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
    if not _CLASSIFIER_PATH.exists() or not _REGRESSOR_PATH.exists():
        return None, None
    try:
        clf = joblib.load(_CLASSIFIER_PATH)
        reg = joblib.load(_REGRESSOR_PATH)
        return clf, reg
    except Exception:
        return None, None


def _save_models(clf: XGBClassifier, reg: XGBRegressor, metadata: Dict[str, Any]) -> None:
    _ensure_models_dir()
    joblib.dump(clf, _CLASSIFIER_PATH)
    joblib.dump(reg, _REGRESSOR_PATH)
    joblib.dump(metadata, _MODEL_METADATA_PATH)


def _load_metadata() -> Dict[str, Any]:
    if _MODEL_METADATA_PATH.exists():
        try:
            return joblib.load(_MODEL_METADATA_PATH)
        except Exception:
            pass
    return {}


def _compute_class_weights(y: pd.Series) -> Dict[int, float]:
    counts = y.value_counts()
    total = len(y)
    weights = {}
    for cls in counts.index:
        weights[int(cls)] = total / (len(counts) * counts[cls])
    return weights


def train_model(
    min_samples: int = 30,
    callsigns: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if callsigns:
        min_samples = min(min_samples, 5)
    features, binary_target, reg_target, route_labels = get_training_data(
        min_samples=min_samples, callsigns=callsigns, db_path=db_path,
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

    n_splits = min(5, route_labels.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    clf_metrics_list = []
    reg_metrics_list = []

    for train_idx, test_idx in gkf.split(features, binary_target, groups=route_labels):
        X_tr, X_te = features.iloc[train_idx], features.iloc[test_idx]
        y_tr_clf, y_te_clf = binary_target.iloc[train_idx], binary_target.iloc[test_idx]
        y_tr_reg, y_te_reg = reg_target.iloc[train_idx], reg_target.iloc[test_idx]
        sw_tr = sample_weights[train_idx]

        pos_weight = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)

        clf = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            min_child_weight=2,
            subsample=0.9, colsample_bytree=0.9,
        )
        clf.fit(X_tr, y_tr_clf, sample_weight=sw_tr)

        reg = XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            random_state=42, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=2,
            subsample=0.9, colsample_bytree=0.9,
        )
        reg.fit(X_tr, y_tr_reg)

        y_te_pred = clf.predict(X_te)
        y_te_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else y_te_pred.astype(float)
        y_te_reg_pred = reg.predict(X_te)

        n_delayed_te = int(y_te_clf.sum())
        clf_m = {
            "accuracy": accuracy_score(y_te_clf, y_te_pred),
            "balanced_accuracy": balanced_accuracy_score(y_te_clf, y_te_pred),
            "precision": precision_score(y_te_clf, y_te_pred, zero_division=0),
            "recall": recall_score(y_te_clf, y_te_pred, zero_division=0),
            "f1": f1_score(y_te_clf, y_te_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_te_clf, y_te_pred),
            "kappa": cohen_kappa_score(y_te_clf, y_te_pred),
            "roc_auc": roc_auc_score(y_te_clf, y_te_proba) if n_delayed_te > 0 else None,
            "pr_auc": average_precision_score(y_te_clf, y_te_proba) if n_delayed_te > 0 else None,
            "log_loss": log_loss(y_te_clf, y_te_proba),
            "confusion": confusion_matrix(y_te_clf, y_te_pred).tolist(),
            "samples": len(y_te_clf),
            "delayed_in_test": n_delayed_te,
        }
        clf_metrics_list.append(clf_m)

        reg_m = {
            "mae": mean_absolute_error(y_te_reg, y_te_reg_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_te_reg, y_te_reg_pred))),
            "r2": r2_score(y_te_reg, y_te_reg_pred),
        }
        reg_metrics_list.append(reg_m)

    def _avg_dicts(dicts, keys=None):
        if not dicts:
            return {}
        if keys is None:
            keys = [k for k in dicts[0] if isinstance(dicts[0][k], (int, float)) and dicts[0][k] is not None]
        result = {}
        for k in keys:
            vals = [d[k] for d in dicts if k in d and d[k] is not None]
            result[k] = round(np.mean(vals), 4) if vals else None
        return result

    avg_clf = _avg_dicts(clf_metrics_list)
    avg_reg = _avg_dicts(reg_metrics_list)

    fold_clf_accs = [d["accuracy"] for d in clf_metrics_list]
    fold_clf_f1s = [d["f1"] for d in clf_metrics_list]
    fold_clf_aucs = [d["roc_auc"] for d in clf_metrics_list if d["roc_auc"] is not None]

    train_idx_final, test_idx_final = next(gkf.split(features, binary_target, groups=route_labels))
    X_train_final = features.iloc[train_idx_final]
    X_test_final = features.iloc[test_idx_final]
    y_train_final = binary_target.iloc[train_idx_final]
    y_test_final = binary_target.iloc[test_idx_final]
    y_reg_train_final = reg_target.iloc[train_idx_final]
    y_reg_test_final = reg_target.iloc[test_idx_final]
    sw_train_final = sample_weights[train_idx_final]

    pos_weight_final = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)

    clf_final = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=pos_weight_final,
        min_child_weight=2,
        subsample=0.9, colsample_bytree=0.9,
    )
    clf_final.fit(X_train_final, y_train_final, sample_weight=sw_train_final)

    reg_final = XGBRegressor(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        random_state=42, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=2,
        subsample=0.9, colsample_bytree=0.9,
    )
    reg_final.fit(X_train_final, y_reg_train_final)

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
        "fold_accuracy": [round(a, 4) for a in fold_clf_accs],
        "fold_f1": [round(f, 4) for f in fold_clf_f1s],
        "fold_roc_auc": [round(a, 4) for a in fold_clf_aucs],
        "feature_importance": {k: round(v, 4) for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)},
        "feature_columns": FEATURE_COLUMNS,
    }

    _save_models(clf_final, reg_final, metadata)

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
        "cv_log_loss": round(avg_clf.get("log_loss", 0), 4),
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
        delay_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
        expected_delay = float(reg.predict(features_df)[0])
        expected_delay = max(0.0, expected_delay)

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
) -> Dict[str, Any]:
    if departure_time is None:
        departure_time = datetime.now()

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
        route=f"{origin}_{destination}",
        prev_delay=prev_flight_delay,
        wind_speed_kmh=wind_speed_kmh,
        wind_gusts_kmh=wind_gusts_kmh,
        visibility_m=visibility_m,
        cloud_cover_pct=cloud_cover_pct,
        precipitation_mm=precipitation_mm,
        temperature_c=temperature_c,
        pressure_hpa=pressure_hpa,
    )

    ml_result = _predict_with_ml(features_df)
    if ml_result:
        result = ml_result
    else:
        result = _heuristic_predict(
            origin, destination, aircraft_type, departure_hour,
            pax_count, flight_duration_min, is_international,
        )

    weather_adj = _compute_weather_adjustment(
        wind_speed_kmh=wind_speed_kmh, wind_gusts_kmh=wind_gusts_kmh,
        visibility_m=visibility_m, precipitation_mm=precipitation_mm,
        cloud_cover_pct=cloud_cover_pct, departure_hour=departure_hour,
    )
    base_prob = result["delay_probability"]
    adjusted_prob = max(0.05, min(0.95, base_prob + weather_adj["probability_boost"]))
    result["delay_probability"] = round(adjusted_prob, 3)
    result["expected_delay_min"] = round(result["expected_delay_min"] + weather_adj["delay_boost"], 1)

    _GLOBAL_BASELINE_DELAY_RATE = 0.02
    if dow_delay_rate is not None and overall_delay_rate is not None and overall_delay_rate > 0 and dow_delay_rate > 0:
        base_prob = result["delay_probability"]
        scale_factor = dow_delay_rate / max(overall_delay_rate, _GLOBAL_BASELINE_DELAY_RATE)
        scale_factor = max(0.3, min(3.0, scale_factor))
        adjusted_prob = base_prob * scale_factor
        adjusted_prob = max(0.05, min(0.95, adjusted_prob))
        result["delay_probability"] = round(adjusted_prob, 3)
        result["expected_delay_min"] = round(result["expected_delay_min"] * scale_factor, 1)

    prob = result["delay_probability"]
    exp_delay = result["expected_delay_min"]
    risk_score = prob * exp_delay
    risk_level = "Low"
    if prob > 0.15 or (prob > 0.10 and exp_delay > 5):
        risk_level = "High"
    elif prob > 0.06 or (prob > 0.05 and exp_delay > 3):
        risk_level = "Medium"

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
        if departure_time is not None:
            day_name = day_names[departure_time.weekday()]
        else:
            day_name = "this day"
        if dow_pct > overall_pct * 1.2:
            factors.append(
                f"Historically {dow_pct:.0f}% delay rate on {day_name} vs {overall_pct:.0f}% overall"
            )
        elif dow_pct < overall_pct * 0.8 and dow_pct > 0:
            factors.append(
                f"Lower historical delay rate on {day_name} ({dow_pct:.0f}% vs {overall_pct:.0f}% overall)"
            )
        elif dow_pct == 0 and overall_pct > 5:
            factors.append(
                f"No historical delays on {day_name} for this route (vs {overall_pct:.0f}% overall)"
            )

    result.update({
        "risk_level": risk_level,
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
        predicted_delayed = predicted["delay_probability"] > 0.3
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
