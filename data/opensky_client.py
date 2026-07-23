from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


_CREDENTIALS_PATH = Path(__file__).parent.parent / "credentials.json"
_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
_BASE_URL = "https://opensky-network.org/api"

# BLR (Kempegowda International Airport) bounding box
BLR_LAT = 13.1986
BLR_LON = 77.7066
BLR_BBOX = (12.5, 77.0, 13.9, 78.4)  # lamin, lomin, lamax, lomax


class OpenSkyTokenManager:
    def __init__(self, credentials_path: Optional[Path] = None):
        path = credentials_path or _CREDENTIALS_PATH
        with open(path, "r") as f:
            creds = json.load(f)
        self.client_id = creds["clientId"]
        self.client_secret = creds["clientSecret"]
        self._token: Optional[str] = None
        self._expires_at: Optional[datetime] = None

    def get_token(self) -> str:
        if self._token and self._expires_at and datetime.now(timezone.utc) < self._expires_at:
            return self._token
        return self._refresh()

    def _refresh(self) -> str:
        resp = requests.post(
            _TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        expires_in = data.get("expires_in", 1800)
        self._expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 30)
        return self._token

    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.get_token()}"}


class OpenSkyClient:
    def __init__(self, credentials_path: Optional[Path] = None):
        self.token_mgr = OpenSkyTokenManager(credentials_path)
        self._last_request_time: float = 0.0
        self._min_interval: float = 10.5  # seconds between requests
        self.credits_remaining: Optional[int] = None

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Any:
        self._rate_limit()
        url = f"{_BASE_URL}{endpoint}"
        resp = requests.get(url, params=params, headers=self.token_mgr.headers(), timeout=30)

        remaining = resp.headers.get("X-Rate-Limit-Remaining")
        if remaining is not None:
            try:
                self.credits_remaining = int(remaining)
            except (ValueError, TypeError):
                pass

        if resp.status_code == 404:
            return []

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("X-Rate-Limit-Retry-After-Seconds", 60))
            time.sleep(retry_after)
            return self._get(endpoint, params)

        if resp.status_code == 401:
            self.token_mgr._token = None
            self.token_mgr._expires_at = None
            return self._get(endpoint, params)

        resp.raise_for_status()
        return resp.json()

    def get_departures(
        self,
        airport: str = "VOBL",
        begin: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if begin is None or end is None:
            now = int(time.time())
            end = end or now
            begin = begin or (end - 86400)  # last 24h
        result = self._get("/flights/departure", {
            "airport": airport,
            "begin": begin,
            "end": end,
        })
        return result if isinstance(result, list) else []

    def get_arrivals(
        self,
        airport: str = "VOBL",
        begin: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if begin is None or end is None:
            now = int(time.time())
            end = end or now
            begin = begin or (end - 86400)
        result = self._get("/flights/arrival", {
            "airport": airport,
            "begin": begin,
            "end": end,
        })
        return result if isinstance(result, list) else []

    def get_live_states(
        self,
        bbox: Tuple[float, float, float, float] = BLR_BBOX,
    ) -> List[Dict[str, Any]]:
        lamin, lomin, lamax, lomax = bbox
        result = self._get("/states/all", {
            "lamin": lamin,
            "lomin": lomin,
            "lamax": lamax,
            "lomax": lomax,
        })
        raw_states = result.get("states") or []
        fields = [
            "icao24", "callsign", "origin_country", "time_position",
            "last_contact", "longitude", "latitude", "baro_altitude",
            "on_ground", "velocity", "true_track", "vertical_rate",
            "sensors", "geo_altitude", "squawk", "spi",
            "position_source", "category",
        ]
        parsed = []
        for row in raw_states:
            if len(row) >= len(fields):
                parsed.append(dict(zip(fields, row[:len(fields)])))
            else:
                parsed.append(dict(zip(fields[:len(row)], row)))
        return parsed

    def get_track(self, icao24: str, time_secs: int = 0) -> Optional[Dict[str, Any]]:
        return self._get("/tracks/all", {"icao24": icao24, "time": time_secs})

    def get_flights_by_aircraft(
        self,
        icao24: str,
        begin: int,
        end: int,
    ) -> List[Dict[str, Any]]:
        result = self._get("/flights/aircraft", {
            "icao24": icao24.lower(),
            "begin": begin,
            "end": end,
        })
        return result if isinstance(result, list) else []


def utc_day_range(date: datetime) -> Tuple[int, int]:
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


def get_blr_bounding_box() -> Tuple[float, float, float, float]:
    return BLR_BBOX
