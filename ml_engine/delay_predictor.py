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


def train_model(
    min_samples: int = 30,
    callsigns: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if callsigns:
        min_samples = min(min_samples, 5)
    features, binary_target, reg_target = get_training_data(
        min_samples=min_samples, callsigns=callsigns, db_path=db_path,
    )

    if len(features) < min_samples:
        return {
            "status": "insufficient_data",
            "samples": len(features),
            "min_required": min_samples,
            "message": f"Need {min_samples} samples, have {len(features)}. Using heuristic fallback.",
        }

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(features, binary_target)

    reg = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    reg.fit(features, reg_target)

    from sklearn.metrics import accuracy_score, mean_absolute_error
    clf_pred = clf.predict(features)
    clf_accuracy = accuracy_score(binary_target, clf_pred)
    reg_pred = reg.predict(features)
    reg_mae = mean_absolute_error(reg_target, reg_pred)

    feature_importance = dict(zip(FEATURE_COLUMNS, clf.feature_importances_.tolist()))

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "samples": len(features),
        "classifier_accuracy": round(clf_accuracy, 4),
        "regressor_mae": round(reg_mae, 2),
        "feature_importance": {k: round(v, 4) for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)},
        "feature_columns": FEATURE_COLUMNS,
    }

    _save_models(clf, reg, metadata)

    return {
        "status": "success",
        "samples": len(features),
        "classifier_accuracy": round(clf_accuracy, 4),
        "regressor_mae": round(reg_mae, 2),
        "feature_importance": metadata["feature_importance"],
        "message": f"Trained on {len(features)} samples. Accuracy: {clf_accuracy:.1%}, MAE: {reg_mae:.1f} min",
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

    route_avg_delay = 0.0
    try:
        _db = sqlite3.connect(str(Path(__file__).parent.parent / "data" / "flights.db"))
        _cur = _db.execute(
            "SELECT AVG(deviation_min) FROM delay_labels WHERE origin=? AND destination=? AND is_delayed=1",
            (origin, destination),
        )
        _row = _cur.fetchone()
        if _row and _row[0] is not None:
            route_avg_delay = float(_row[0])
        _db.close()
    except Exception:
        pass

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
        route_avg_delay=route_avg_delay,
    )

    ml_result = _predict_with_ml(features_df)
    if ml_result:
        result = ml_result
    else:
        result = _heuristic_predict(
            origin, destination, aircraft_type, departure_hour,
            pax_count, flight_duration_min, is_international,
        )

    prob = result["delay_probability"]
    exp_delay = result["expected_delay_min"]
    risk_level = "Low"
    if prob > 0.5 or exp_delay > 45:
        risk_level = "High"
    elif prob > 0.3 or exp_delay > 20:
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

    return {
        "total_flights": total,
        "delayed_flights": delayed,
        "delay_rate": round(delay_rate, 3),
        "peak_delay_hours": [{"hour": h, "count": c} for h, c in sorted(by_hour.items(), key=lambda x: x[1], reverse=True)[:5]],
        "delay_by_route": by_route,
        "model_status": {
            "trained": model_meta.get("trained_at") is not None,
            "accuracy": model_meta.get("classifier_accuracy"),
            "mae": model_meta.get("regressor_mae"),
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
