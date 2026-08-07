"""Central tuning knobs for the delay-prediction ML pipeline and the crew solver.

Values here are the single source of truth for these knobs. Modules that used to
define them as local constants now import from here; defaults are byte-for-byte
identical to the previous hard-coded values, so behaviour is unchanged.
"""

# --- Delay definition -----------------------------------------------------
# A flight counts as "delayed" when it departs more than this many minutes late.
DELAY_THRESHOLD_MIN = 15.0

# --- Weekday weighting ----------------------------------------------------
WEEKDAY_WEIGHT_DEFAULT = 0.75
WEEKDAY_SAMPLE_FOR_FULL_CONFIDENCE = 8

# --- Risk bands (forecast display only) -----------------------------------
RISK_HIGH_PROB = 0.37
RISK_MEDIUM_PROB = 0.30
RISK_MEDIUM_PROB_BY_DOW = {
    0: 0.30,  # Mon
    1: 0.32,  # Tue
    2: 0.28,  # Wed
    3: 0.28,  # Thu
    4: 0.30,  # Fri
    5: 0.30,  # Sat
    6: 0.32,  # Sun
}

# Baseline weather used to compute relative wind/precip/visibility ratios.
WEATHER_BASELINE = {
    "wind_kmh": 21.0,
    "precip_mm": 0.1,
    "vis_m": 10000.0,
    "cloud_pct": 88.5,
}

# Blend weight applied to the recent-history prior rate vs the ML probability.
PREDICT_BLEND_WEIGHT = 0.7

# --- Online prediction-correction layer -----------------------------------
CORR_WINDOW_DAYS = 7           # look back over this many days of actuals
CORR_MIN_SAMPLES = 20          # min matched pred->actual pairs per route to adjust
CORR_DEADBAND_PROB = 0.03      # |drift| above this (3 pts) triggers a prob update
CORR_DEADBAND_MIN = 10.0       # |residual| above this (10 min) triggers a minutes update
CORR_LEARNING_RATE = 0.2       # move this fraction of the gap per update (gradual)
CORR_MAX_LOGIT_BIAS = 0.5      # bounds for the log-odds bias
CORR_MAX_MIN_BIAS = 30.0       # bounds for the minutes bias
CORR_CACHE_TTL_S = 300         # how long a route's bias row is cached in-process

# --- Solver (PuLP crew assignment) defaults -------------------------------
SOLVER_SCENARIO_FLIGHT_HOURS = 2.0
SOLVER_SCENARIO_IS_NIGHT = False
SOLVER_REQUIRED_COUNTS = {
    "Captain": 1,
    "FO": 1,
    "CabinCrew": 2,
    "RampAgent": 1,
    "BaggageHandler": 1,
    "CabinCleaner": 1,
    "CheckinAgent": 1,
    "SecurityAgent": 1,
}
