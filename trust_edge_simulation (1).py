"""
================================================================================
TRUST-EDGE : PHYSICS-INFORMED COLD-CHAIN THERMAL SIMULATION
================================================================================

  *** ALL DATA PRODUCED BY THIS SCRIPT IS SIMULATED / MODEL-GENERATED. ***
  *** IT IS NOT A PHYSICAL SENSOR MEASUREMENT AND MUST NEVER BE        ***
  *** PRESENTED AS EXPERIMENTAL EVIDENCE.                              ***

Purpose
-------
Establish a reproducible, physics-informed BASELINE for the Trust-Edge
pipeline:

        Sense -> Predict -> Estimate Safe Time Remaining (ESTR)
              -> Decide -> Act -> Log

The same processing chain implemented here will later be fed with REAL
ESP32 / DS18B20 / DHT11 measurements. At that point this simulated CSV is
replaced by the recorded CSV, and the numbers become experimental evidence.

Models used
-----------
1. Plant / ground truth (first-order Newton-style thermal model)

        dTp/dt = (Ta - Tp) / tau

   integrated with the exact zero-order-hold solution over each step:

        Tp(t+dt) = Ta + (Tp(t) - Ta) * exp(-dt / tau)

2. Predictor A - THERMAL (RC / first-order) model
   tau is NOT assumed known. It is estimated online from the measurements:

        tau_hat = (Ta_meas - Tp_meas) / (dTp/dt)_measured
        T_pred(t + H) = Ta + (Tp - Ta) * exp(-H / tau_hat)

3. Predictor B - LINEAR baseline

        T_pred(t + H) = Tp + (dTp/dt) * H

No machine learning is used anywhere in this version, by design.

Author : Trust-Edge project
Python : 3.9+   Requires: numpy, pandas, matplotlib
================================================================================
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")  # headless backend: works on any laptop, no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. GLOBAL CONFIGURATION  (every magic number lives here, nothing is hidden)
# ==============================================================================

DATA_TYPE_TAG = "SIMULATED"          # written into every CSV row
SIM_BANNER = "SIMULATED DATA - MODEL GENERATED, NOT PHYSICAL MEASUREMENT"

# ---- Safety limits -----------------------------------------------------------
# NOTE: 8.0 degC is a PROTOTYPE DEMONSTRATION PARAMETER ONLY.
# In a real deployment it MUST be replaced by the validated, product-specific
# storage limit taken from the manufacturer / pharmacopoeia storage label
# (for example the upper limit of a 2-8 degC vaccine storage window).
SAFETY_THRESHOLD_C = 8.0             # upper safety limit used for ESTR
NOMINAL_SETPOINT_C = 5.0             # nominal cold-chain setpoint (risk baseline)

# ---- Sampling ----------------------------------------------------------------
SAMPLE_INTERVAL_S = 10.0             # 10 s sampling (realistic for a logger)
DERIV_WINDOW_S = 600.0               # 10 min least-squares window for dT/dt
DERIV_MIN_POINTS = 5                 # minimum points before a slope is trusted
SLOPE_EMA_ALPHA = 0.30               # extra smoothing on the slope estimate

# ---- Prediction --------------------------------------------------------------
PREDICTION_HORIZON_MIN = 10.0        # predict this far ahead
TAU_EMA_ALPHA = 0.25                 # smoothing of the online tau estimate
TAU_EMA_ALPHA_FAST = 0.60            # faster adaptation when the thermal regime changes
TAU_REGIME_CHANGE_RATIO = 2.0        # candidate differs by >2x -> regime change detected
TAU_MIN_S = 300.0                    # sanity clip on tau_hat (5 min)
TAU_MAX_S = 400000.0                 # sanity clip on tau_hat (~111 h)
MIN_GRADIENT_C = 0.50                # |Ta - Tp| below this -> tau not observable
MIN_SIGNIFICANT_TREND_C_PER_MIN = 0.015   # slope noise floor gate (see README)
MIN_RATE_C_PER_S = MIN_SIGNIFICANT_TREND_C_PER_MIN / 60.0   # same gate, per second
ESTR_CAP_MIN = 240.0                 # 4 h reporting horizon; >= cap == "no crossing predicted"
ESTR_BAND_FLOOR_MIN = 1.0            # minimum +/- width reported around ESTR (minutes)

# ---- Sensor imperfection (matches the parts on the Trust-Edge bench) ---------
DS18B20_NOISE_STD_C = 0.05           # gaussian noise, degC
DS18B20_RESOLUTION_C = 0.0625        # 12-bit DS18B20 quantisation step
DHT11_NOISE_STD_RH = 0.40            # gaussian noise, %RH
DHT11_RESOLUTION_RH = 1.0            # DHT11 reports whole %RH

# ---- Risk model (transparent, explainable, weighted sum) --------------------
RISK_WEIGHTS = {
    "proximity": 0.40,               # how close the payload already is to the limit
    "temp_trend": 0.30,              # how fast it is moving towards the limit
    "humidity_trend": 0.10,          # corroborating environmental evidence ONLY
    "lid": 0.20,                     # enclosure integrity signal
}
TREND_REF_C_PER_MIN = 0.10           # dT/dt that saturates the trend sub-score
HUM_TREND_REF_RH_PER_MIN = 0.50      # dH/dt that saturates the humidity sub-score

RISK_CRITICAL_MIN_SCORE = 50.0       # score must ALSO be this high before CRITICAL/actuation
RISK_HIGH_SCORE = 60.0
RISK_MEDIUM_SCORE = 30.0
ESTR_CRITICAL_MIN = 10.0             # ESTR below this + elevated score -> CRITICAL
ESTR_HIGH_MIN = 30.0
ESTR_MEDIUM_MIN = 60.0

# ---- Actuation (one-shot emergency release) ---------------------------------
ACTUATION_PERSISTENCE_S = 60.0       # CRITICAL must persist this long before firing

# ---- Reproducibility ---------------------------------------------------------
MASTER_SEED = 20260911               # deterministic: same seed -> same CSV

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# 2. EXPERIMENT DEFINITIONS
# ==============================================================================

@dataclass
class ExperimentConfig:
    """One simulated cold-chain run."""
    experiment_id: int
    name: str
    thermal_condition: str
    seed: int
    duration_min: float
    start_payload_c: float
    start_humidity_rh: float

    # ambient temperature profile: linear ramp from -> to, starting at ramp_start_min
    ambient_start_c: float
    ambient_end_c: float
    ambient_ramp_start_min: float

    # enclosure
    lid_open_min: float | None       # None = lid stays CLOSED for the whole run
    tau_closed_s: float              # effective thermal time constant, lid closed
    tau_open_s: float                # effective thermal time constant, lid open

    # humidity behaviour
    humidity_closed_rh: float
    humidity_ambient_rh: float
    tau_hum_closed_s: float = 3600.0
    tau_hum_open_s: float = 600.0

    notes: str = ""


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        experiment_id=1,
        name="EXP1_NORMAL_STABLE",
        thermal_condition="normal_stable_lid_closed",
        seed=MASTER_SEED + 1,
        duration_min=120.0,
        start_payload_c=5.0,
        start_humidity_rh=45.0,
        ambient_start_c=25.0,
        ambient_end_c=25.6,          # slow room drift only
        ambient_ramp_start_min=0.0,
        lid_open_min=None,           # never opened
        tau_closed_s=144000.0,       # ~40 h : good insulated cold box, lid sealed
        tau_open_s=144000.0,
        humidity_closed_rh=45.0,
        humidity_ambient_rh=62.0,
        notes="Baseline. Payload drifts only ~1 degC in 2 h. Risk must stay LOW.",
    ),
    ExperimentConfig(
        experiment_id=2,
        name="EXP2_MODERATE_DISTURBANCE",
        thermal_condition="moderate_disturbance_lid_opened",
        seed=MASTER_SEED + 2,
        duration_min=120.0,
        start_payload_c=5.0,
        start_humidity_rh=45.0,
        ambient_start_c=25.0,
        ambient_end_c=28.5,
        ambient_ramp_start_min=20.0,
        lid_open_min=20.0,           # lid opened at t = 20 min and left ajar
        tau_closed_s=144000.0,
        tau_open_s=54000.0,          # ~15 h : partial opening, restricted airflow
        humidity_closed_rh=45.0,
        humidity_ambient_rh=64.0,
        notes="Lid ajar + warming room. Payload approaches but does NOT cross 8 degC.",
    ),
    ExperimentConfig(
        experiment_id=3,
        name="EXP3_SEVERE_DISTURBANCE",
        thermal_condition="severe_disturbance_lid_open",
        seed=MASTER_SEED + 3,
        duration_min=120.0,
        start_payload_c=5.0,
        start_humidity_rh=45.0,
        ambient_start_c=25.0,
        ambient_end_c=34.0,
        ambient_ramp_start_min=15.0,
        lid_open_min=15.0,           # lid fully open at t = 15 min
        tau_closed_s=144000.0,
        tau_open_s=14400.0,          # ~4 h : lid fully open, free convection
        humidity_closed_rh=45.0,
        humidity_ambient_rh=68.0,
        notes="Fully open lid + hot loading bay. Payload crosses 8 degC -> CRITICAL.",
    ),
    ExperimentConfig(
        experiment_id=4,
        name="EXP4_REFRIGERATION_FAILURE",
        thermal_condition="refrigeration_failure_low_ambient",
        seed=MASTER_SEED + 4,
        duration_min=120.0,
        start_payload_c=5.0,
        start_humidity_rh=45.0,
        ambient_start_c=12.0,        # a failed chiller room, only slightly warm
        ambient_end_c=12.0,
        ambient_ramp_start_min=0.0,
        lid_open_min=10.0,
        tau_closed_s=144000.0,
        tau_open_s=7200.0,           # ~2 h : payload ends up close to ambient
        humidity_closed_rh=45.0,
        humidity_ambient_rh=55.0,
        notes=("Diagnostic case. Ambient (12 C) is only 4 C above the 8 C limit, so "
               "the exponential approach curves strongly and the linear baseline "
               "should visibly under-estimate the remaining safe time."),
    ),
]

SIM_START_TIME = datetime(2026, 9, 11, 9, 0, 0)


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def quantise(value: float, step: float) -> float:
    """Emulate finite ADC / sensor resolution."""
    return round(value / step) * step


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def ambient_profile(cfg: ExperimentConfig, t_s: float) -> float:
    """Piecewise-linear ambient temperature profile, degC."""
    t_min = t_s / 60.0
    if t_min <= cfg.ambient_ramp_start_min:
        return cfg.ambient_start_c
    span = max(1e-6, cfg.duration_min - cfg.ambient_ramp_start_min)
    frac = min(1.0, (t_min - cfg.ambient_ramp_start_min) / span)
    return cfg.ambient_start_c + frac * (cfg.ambient_end_c - cfg.ambient_start_c)


def lid_is_open(cfg: ExperimentConfig, t_s: float) -> bool:
    if cfg.lid_open_min is None:
        return False
    return (t_s / 60.0) >= cfg.lid_open_min


def rolling_slope_per_min(values: np.ndarray, times_s: np.ndarray,
                          window_s: float, min_points: int) -> np.ndarray:
    """
    Least-squares slope over a trailing time window.

    Returns slope in units-per-MINUTE. This is exactly what firmware would do
    on the ESP32: keep the last N samples in a ring buffer and fit a line.
    """
    n = len(values)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        t_end = times_s[i]
        j = i
        while j > 0 and (t_end - times_s[j - 1]) <= window_s:
            j -= 1
        seg_t = times_s[j:i + 1]
        seg_v = values[j:i + 1]
        if len(seg_t) < min_points:
            out[i] = 0.0
            continue
        # slope in units per second -> per minute
        slope_per_s = np.polyfit(seg_t, seg_v, 1)[0]
        out[i] = slope_per_s * 60.0
    return out


def ema(series: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average (single-pole IIR filter)."""
    out = np.zeros_like(series, dtype=float)
    acc = series[0] if len(series) else 0.0
    for i, v in enumerate(series):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


# ==============================================================================
# 4. PLANT SIMULATION  (ground truth generation)
# ==============================================================================

def simulate_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Generate one experiment: ground truth physics + simulated sensor readings.
    Nothing here is a manually invented temperature curve - every value comes
    out of the first-order thermal ODE.
    """
    rng = np.random.default_rng(cfg.seed)

    n_samples = int(round(cfg.duration_min * 60.0 / SAMPLE_INTERVAL_S)) + 1
    times_s = np.arange(n_samples, dtype=float) * SAMPLE_INTERVAL_S

    tp_true = np.zeros(n_samples)
    ta_true = np.zeros(n_samples)
    hum_true = np.zeros(n_samples)
    lid_open = np.zeros(n_samples, dtype=bool)
    tau_used = np.zeros(n_samples)

    tp = cfg.start_payload_c
    hum = cfg.start_humidity_rh

    for i, t in enumerate(times_s):
        ta = ambient_profile(cfg, t)
        is_open = lid_is_open(cfg, t)

        tp_true[i] = tp
        ta_true[i] = ta
        hum_true[i] = hum
        lid_open[i] = is_open

        tau = cfg.tau_open_s if is_open else cfg.tau_closed_s
        tau_h = cfg.tau_hum_open_s if is_open else cfg.tau_hum_closed_s
        tau_used[i] = tau

        # ---- exact zero-order-hold update of dTp/dt = (Ta - Tp)/tau ----------
        decay = math.exp(-SAMPLE_INTERVAL_S / tau)
        tp = ta + (tp - ta) * decay

        # ---- humidity relaxes towards its target the same way ---------------
        h_target = cfg.humidity_ambient_rh if is_open else cfg.humidity_closed_rh
        decay_h = math.exp(-SAMPLE_INTERVAL_S / tau_h)
        hum = h_target + (hum - h_target) * decay_h

    # ---- simulated sensor readings: noise + finite resolution ---------------
    tp_meas = np.array([
        quantise(v + rng.normal(0.0, DS18B20_NOISE_STD_C), DS18B20_RESOLUTION_C)
        for v in tp_true
    ])
    ta_meas = np.array([
        quantise(v + rng.normal(0.0, DS18B20_NOISE_STD_C), DS18B20_RESOLUTION_C)
        for v in ta_true
    ])
    hum_meas = np.array([
        float(np.clip(quantise(v + rng.normal(0.0, DHT11_NOISE_STD_RH),
                               DHT11_RESOLUTION_RH), 0.0, 100.0))
        for v in hum_true
    ])

    df = pd.DataFrame({
        "time_s": times_s,
        "time_minutes": times_s / 60.0,
        "experiment_id": cfg.experiment_id,
        "thermal_condition": cfg.thermal_condition,
        "payload_temperature": np.round(tp_meas, 4),
        "ambient_temperature": np.round(ta_meas, 4),
        "humidity": np.round(hum_meas, 1),
        "lid_status": np.where(lid_open, "OPEN", "CLOSED"),
        "ground_truth_payload_temperature": np.round(tp_true, 4),
        "ground_truth_ambient_temperature": np.round(ta_true, 4),
        "ground_truth_tau_s": tau_used,
    })
    df["timestamp"] = [SIM_START_TIME + timedelta(seconds=float(t)) for t in times_s]
    return df


# ==============================================================================
# 5. EDGE PIPELINE  (this is the part that will later run on ESP32 / PYNQ-Z2)
# ==============================================================================

def process_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs on the MEASURED columns only (payload_temperature, ambient_temperature,
    humidity, lid_status). Ground-truth columns are used strictly for scoring
    afterwards, never inside the estimator.
    """
    df = df.copy().reset_index(drop=True)
    t_s = df["time_s"].to_numpy()
    tp = df["payload_temperature"].to_numpy()
    ta = df["ambient_temperature"].to_numpy()
    hum = df["humidity"].to_numpy()
    lid_open = (df["lid_status"] == "OPEN").to_numpy()
    n = len(df)

    # ---- 5.1 derivatives from the measured signals --------------------------
    dT_dt = ema(rolling_slope_per_min(tp, t_s, DERIV_WINDOW_S, DERIV_MIN_POINTS),
                SLOPE_EMA_ALPHA)                        # degC / min
    dH_dt = ema(rolling_slope_per_min(hum, t_s, DERIV_WINDOW_S, DERIV_MIN_POINTS),
                SLOPE_EMA_ALPHA)                        # %RH / min

    # ---- 5.1b estimator warm-up ---------------------------------------------
    # Real firmware cannot report a trend until its ring buffer is full. During
    # the first DERIV_WINDOW_S seconds the slope is declared INVALID, so the
    # pipeline reports "no prediction available" instead of extrapolating noise.
    warmup = t_s < DERIV_WINDOW_S
    dT_dt[warmup] = 0.0
    dH_dt[warmup] = 0.0
    estimator_status = np.where(warmup, "WARMUP", "OK")

    # ---- 5.2 online estimate of the thermal time constant -------------------
    #        tau_hat = (Ta - Tp) / (dTp/dt)      [both measured]
    tau_hat = np.zeros(n)
    tau_acc = TAU_MAX_S
    for i in range(n):
        gradient = ta[i] - tp[i]
        rate_per_s = dT_dt[i] / 60.0
        if gradient > MIN_GRADIENT_C and rate_per_s > MIN_RATE_C_PER_S:
            candidate = gradient / rate_per_s
            candidate = float(np.clip(candidate, TAU_MIN_S, TAU_MAX_S))
            # If the enclosure regime changes (for example the lid is opened) the
            # true tau changes by a large factor. Detect that and adapt faster,
            # otherwise filter gently to reject sensor noise.
            ratio = max(candidate / tau_acc, tau_acc / candidate)
            alpha = TAU_EMA_ALPHA_FAST if ratio > TAU_REGIME_CHANGE_RATIO else TAU_EMA_ALPHA
            tau_acc = alpha * candidate + (1.0 - alpha) * tau_acc
        tau_hat[i] = tau_acc

    # ---- 5.3 predictions at the horizon -------------------------------------
    horizon_s = PREDICTION_HORIZON_MIN * 60.0
    pred_thermal = ta + (tp - ta) * np.exp(-horizon_s / tau_hat)   # RC model
    pred_linear = tp + dT_dt * PREDICTION_HORIZON_MIN              # linear baseline

    # ---- 5.4 ESTR ------------------------------------------------------------
    estr_rc = np.zeros(n)
    estr_lin = np.zeros(n)
    for i in range(n):
        # --- thermal / RC model ESTR:
        #     solve  Ta + (Tp - Ta) e^{-t/tau} = T_threshold
        #     ->     t = tau * ln( (Ta - Tp) / (Ta - T_threshold) )
        if tp[i] >= SAFETY_THRESHOLD_C:
            estr_rc[i] = 0.0
        elif ta[i] <= SAFETY_THRESHOLD_C + 1e-6:
            estr_rc[i] = ESTR_CAP_MIN          # ambient cannot push it past the limit
        elif dT_dt[i] <= MIN_SIGNIFICANT_TREND_C_PER_MIN:
            # measured trend is inside the slope noise floor -> refuse to
            # extrapolate a crossing time from noise (same gate for both models)
            estr_rc[i] = ESTR_CAP_MIN
        else:
            ratio = (ta[i] - tp[i]) / (ta[i] - SAFETY_THRESHOLD_C)
            estr_rc[i] = min(ESTR_CAP_MIN, max(0.0, tau_hat[i] * math.log(ratio) / 60.0))

        # --- linear baseline ESTR: (T_threshold - Tp) / (dT/dt)
        if tp[i] >= SAFETY_THRESHOLD_C:
            estr_lin[i] = 0.0
        elif dT_dt[i] <= MIN_SIGNIFICANT_TREND_C_PER_MIN:
            estr_lin[i] = ESTR_CAP_MIN
        else:
            estr_lin[i] = min(ESTR_CAP_MIN,
                              (SAFETY_THRESHOLD_C - tp[i]) / dT_dt[i])

    # ---- 5.4b ESTR uncertainty band -----------------------------------------
    # The master Trust-Edge specification requires ESTR to be reported as a
    # RANGE, never as a single exact number. The band here is the trailing
    # short-term spread of the estimate itself (1 standard deviation over the
    # last DERIV_WINDOW_S seconds), floored at ESTR_BAND_FLOOR_MIN.
    band_n = max(2, int(round(DERIV_WINDOW_S / SAMPLE_INTERVAL_S)))
    spread = pd.Series(estr_rc).rolling(band_n, min_periods=2).std().fillna(0.0).to_numpy()
    band = np.maximum(spread, ESTR_BAND_FLOOR_MIN)
    estr_low = np.clip(estr_rc - band, 0.0, ESTR_CAP_MIN)
    estr_high = np.clip(estr_rc + band, 0.0, ESTR_CAP_MIN)

    # ---- 5.5 ground-truth ESTR (only for error scoring, never for decisions) -
    tp_true = df["ground_truth_payload_temperature"].to_numpy()
    estr_true = np.full(n, ESTR_CAP_MIN)
    crossings = np.where(tp_true >= SAFETY_THRESHOLD_C)[0]
    if len(crossings) > 0:
        first_cross = crossings[0]
        t_cross = t_s[first_cross]
        for i in range(n):
            if i >= first_cross:
                estr_true[i] = 0.0
            else:
                estr_true[i] = min(ESTR_CAP_MIN, (t_cross - t_s[i]) / 60.0)

    # ---- 5.6 transparent risk score -----------------------------------------
    band = max(1e-6, SAFETY_THRESHOLD_C - NOMINAL_SETPOINT_C)
    s_prox = np.array([clip01((v - NOMINAL_SETPOINT_C) / band) for v in tp])
    s_trend = np.array([clip01(v / TREND_REF_C_PER_MIN) for v in dT_dt])
    s_hum = np.array([clip01(v / HUM_TREND_REF_RH_PER_MIN) for v in dH_dt])
    s_lid = np.where(lid_open, 1.0, 0.0)

    risk_score = 100.0 * (
        RISK_WEIGHTS["proximity"] * s_prox
        + RISK_WEIGHTS["temp_trend"] * s_trend
        + RISK_WEIGHTS["humidity_trend"] * s_hum
        + RISK_WEIGHTS["lid"] * s_lid
    )

    # ---- 5.7 decision engine -------------------------------------------------
    levels = []
    for i in range(n):
        score = risk_score[i]
        estr = estr_rc[i]
        breached = tp[i] >= SAFETY_THRESHOLD_C
        # CRITICAL is only declared on an actual or imminent PREDICTED breach.
        # A high score alone never fires the actuator (single-indicator protection).
        if breached or (estr <= ESTR_CRITICAL_MIN and score >= RISK_CRITICAL_MIN_SCORE):
            levels.append("CRITICAL")
        elif estr <= ESTR_HIGH_MIN or score >= RISK_HIGH_SCORE:
            levels.append("HIGH")
        elif estr <= ESTR_MEDIUM_MIN or score >= RISK_MEDIUM_SCORE:
            levels.append("MEDIUM")
        else:
            levels.append("LOW")

    # ---- 5.8 one-shot actuation with persistence check ----------------------
    need = int(round(ACTUATION_PERSISTENCE_S / SAMPLE_INTERVAL_S))
    actuator = ["IDLE"] * n
    run = 0
    deployed_at = None
    for i in range(n):
        run = run + 1 if levels[i] == "CRITICAL" else 0
        if deployed_at is None and run >= need:
            deployed_at = i
        actuator[i] = "DEPLOYED" if (deployed_at is not None and i >= deployed_at) else "IDLE"

    df["dT_dt"] = np.round(dT_dt, 5)
    df["dH_dt"] = np.round(dH_dt, 5)
    df["tau_estimated_s"] = np.round(tau_hat, 1)
    df["predicted_temperature"] = np.round(pred_thermal, 4)
    df["predicted_temperature_linear"] = np.round(pred_linear, 4)
    df["ESTR_minutes"] = np.round(estr_rc, 2)
    df["ESTR_low_minutes"] = np.round(estr_low, 2)
    df["ESTR_high_minutes"] = np.round(estr_high, 2)
    df["ESTR_minutes_linear"] = np.round(estr_lin, 2)
    df["ESTR_minutes_ground_truth"] = np.round(estr_true, 2)
    df["score_proximity"] = np.round(s_prox, 4)
    df["score_temp_trend"] = np.round(s_trend, 4)
    df["score_humidity_trend"] = np.round(s_hum, 4)
    df["score_lid"] = np.round(s_lid, 4)
    df["risk_score"] = np.round(risk_score, 2)
    df["risk_level"] = levels
    df["estimator_status"] = estimator_status
    df["actuator_state"] = actuator
    df["data_type"] = DATA_TYPE_TAG
    return df


# ==============================================================================
# 6. MODEL COMPARISON METRICS  (thermal model vs linear baseline)
# ==============================================================================

def compare_models(df: pd.DataFrame) -> dict:
    """
    Compare the horizon-H prediction against the simulated ground truth,
    and compare both ESTR estimators against the true remaining time.
    """
    step = int(round(PREDICTION_HORIZON_MIN * 60.0 / SAMPLE_INTERVAL_S))
    truth = df["ground_truth_payload_temperature"].to_numpy()
    n = len(df)

    valid = np.arange(0, n - step)
    warm = int(round(DERIV_WINDOW_S / SAMPLE_INTERVAL_S))
    valid = valid[valid >= warm]

    future_truth = truth[valid + step]
    err_rc = df["predicted_temperature"].to_numpy()[valid] - future_truth
    err_lin = df["predicted_temperature_linear"].to_numpy()[valid] - future_truth

    rmse_rc = float(np.sqrt(np.mean(err_rc ** 2))) if len(valid) else float("nan")
    rmse_lin = float(np.sqrt(np.mean(err_lin ** 2))) if len(valid) else float("nan")

    # ESTR error is scored ONLY inside a fair evaluation window:
    #   - the disturbance has already started (lid OPEN), because no estimator
    #     can predict a lid opening that has not happened yet;
    #   - a genuine crossing still lies ahead in the ground truth;
    #   - both estimators actually report a finite crossing (not the cap).
    est_true = df["ESTR_minutes_ground_truth"].to_numpy()
    mask = (est_true < ESTR_CAP_MIN - 1e-6) & (est_true > 0.0)
    mask &= (df["lid_status"] == "OPEN").to_numpy()
    # A trailing-window slope estimator provably still contains pre-event samples
    # for DERIV_WINDOW_S seconds after a lid transition. That settling interval is
    # excluded from the metric and reported separately as a known transient.
    lid_open_flag = (df["lid_status"] == "OPEN").to_numpy()
    transition = np.where(np.diff(lid_open_flag.astype(int)) != 0)[0]
    if len(transition):
        t_trans = df["time_s"].to_numpy()[transition[-1] + 1]
        mask &= (df["time_s"].to_numpy() - t_trans) >= DERIV_WINDOW_S
    mask &= df["ESTR_minutes"].to_numpy() < ESTR_CAP_MIN - 1e-6
    mask &= df["ESTR_minutes_linear"].to_numpy() < ESTR_CAP_MIN - 1e-6
    mask[:warm] = False
    if mask.sum() > 0:
        mae_estr_rc = float(np.mean(np.abs(df["ESTR_minutes"].to_numpy()[mask] - est_true[mask])))
        mae_estr_lin = float(np.mean(np.abs(df["ESTR_minutes_linear"].to_numpy()[mask] - est_true[mask])))
    else:
        mae_estr_rc = float("nan")
        mae_estr_lin = float("nan")
    n_estr_samples = int(mask.sum())

    # prediction lead time: first HIGH/CRITICAL vs true threshold crossing
    lead = float("nan")
    crossing_idx = np.where(truth >= SAFETY_THRESHOLD_C)[0]
    warn_idx = np.where(df["risk_level"].isin(["HIGH", "CRITICAL"]).to_numpy())[0]
    if len(crossing_idx) and len(warn_idx):
        lead = float((df["time_minutes"].iloc[crossing_idx[0]]
                      - df["time_minutes"].iloc[warn_idx[0]]))

    return {
        "experiment_id": int(df["experiment_id"].iloc[0]),
        "rmse_thermal_c": rmse_rc,
        "rmse_linear_c": rmse_lin,
        "estr_mae_thermal_min": mae_estr_rc,
        "estr_mae_linear_min": mae_estr_lin,
        "estr_samples": n_estr_samples,
        "lead_time_min": lead,
        "true_crossing_min": (float(df["time_minutes"].iloc[crossing_idx[0]])
                              if len(crossing_idx) else float("nan")),
    }


# ==============================================================================
# 7. PLOTTING
# ==============================================================================

LEVEL_COLORS = {
    "LOW": "#2e7d32",
    "MEDIUM": "#f9a825",
    "HIGH": "#ef6c00",
    "CRITICAL": "#c62828",
}


def _stamp(fig: plt.Figure) -> None:
    fig.text(0.5, 0.005, SIM_BANNER, ha="center", va="bottom",
             fontsize=8, color="#c62828", weight="bold")


def plot_temperature_prediction(all_df: pd.DataFrame, path: str,
                                metrics: list[dict] | None = None) -> None:
    exps = sorted(all_df["experiment_id"].unique())
    fig, axes = plt.subplots(len(exps), 1, figsize=(11, 4.0 * len(exps)), sharex=True)
    if len(exps) == 1:
        axes = [axes]
    step = int(round(PREDICTION_HORIZON_MIN * 60.0 / SAMPLE_INTERVAL_S))

    for ax, eid in zip(axes, exps):
        d = all_df[all_df["experiment_id"] == eid]
        t = d["time_minutes"].to_numpy()
        ax.plot(t, d["ambient_temperature"], color="#9e9e9e", lw=1.0,
                label="Ambient Ta (simulated sensor)")
        if DATA_TYPE_TAG == "SIMULATED":
            ax.plot(t, d["ground_truth_payload_temperature"], color="#1565c0", lw=2.0,
                    label="Payload Tp (model ground truth)")
            ax.plot(t, d["payload_temperature"], color="#42a5f5", lw=0.7, alpha=0.7,
                    label="Payload Tp (simulated sensor + noise)")
        else:
            ax.plot(t, d["payload_temperature"], color="#1565c0", lw=1.4,
                    label="Payload Tp (recorded sensor)")
        # predictions plotted at the time they refer to (t + horizon)
        ax.plot(t + PREDICTION_HORIZON_MIN, d["predicted_temperature"],
                color="#2e7d32", lw=1.4, ls="--",
                label=f"Thermal (RC) prediction, +{PREDICTION_HORIZON_MIN:.0f} min")
        ax.plot(t + PREDICTION_HORIZON_MIN, d["predicted_temperature_linear"],
                color="#c62828", lw=1.2, ls=":",
                label=f"Linear baseline prediction, +{PREDICTION_HORIZON_MIN:.0f} min")
        ax.axhline(SAFETY_THRESHOLD_C, color="#6a1b9a", lw=1.2, ls="-.",
                   label=f"Safety threshold {SAFETY_THRESHOLD_C:.1f} C (prototype parameter)")

        opened = d[d["lid_status"] == "OPEN"]
        if len(opened):
            ax.axvspan(opened["time_minutes"].iloc[0], t[-1],
                       color="#ffcdd2", alpha=0.25, label="Lid OPEN")

        if metrics:
            m = next((x for x in metrics if x["experiment_id"] == eid), None)
            if m:
                ax.text(0.995, 0.03,
                        f"10-min-ahead RMSE   thermal(RC): {m['rmse_thermal_c']:.3f} C"
                        f"   |   linear: {m['rmse_linear_c']:.3f} C",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#90a4ae"))

        cond = d["thermal_condition"].iloc[0]
        ax.set_title(f"Experiment {eid} - {cond}   [{DATA_TYPE_TAG}]", fontsize=11)
        ax.set_ylabel("Temperature (C)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, t[-1])
        ax.legend(fontsize=7, loc="upper left", ncol=2)

    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle("Trust-Edge - Payload temperature and 10-minute-ahead prediction",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _stamp(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _ = step


def plot_estr(all_df: pd.DataFrame, path: str) -> None:
    exps = sorted(all_df["experiment_id"].unique())
    fig, axes = plt.subplots(len(exps), 1, figsize=(11, 3.6 * len(exps)), sharex=True)
    if len(exps) == 1:
        axes = [axes]

    for ax, eid in zip(axes, exps):
        d = all_df[all_df["experiment_id"] == eid]
        t = d["time_minutes"].to_numpy()
        gt_label = ("True remaining time (simulation ground truth)"
                    if DATA_TYPE_TAG == "SIMULATED"
                    else "Actual remaining time (from the recording)")
        ax.plot(t, d["ESTR_minutes_ground_truth"], color="#1565c0", lw=2.2, label=gt_label)
        ax.fill_between(t, d["ESTR_low_minutes"], d["ESTR_high_minutes"],
                        color="#2e7d32", alpha=0.20,
                        label="ESTR uncertainty band (thermal model)")
        ax.plot(t, d["ESTR_minutes"], color="#2e7d32", lw=1.6, ls="--",
                label="ESTR - thermal (RC) model")
        ax.plot(t, d["ESTR_minutes_linear"], color="#c62828", lw=1.4, ls=":",
                label="ESTR - linear baseline")
        ax.axhline(ESTR_CRITICAL_MIN, color="#6a1b9a", lw=1.0, ls="-.",
                   label=f"{ESTR_CRITICAL_MIN:.0f} min action horizon")
        ax.set_ylim(0, ESTR_CAP_MIN * 1.06)
        ax.set_ylabel("ESTR (minutes)")
        ax.set_title(f"Experiment {eid} - {d['thermal_condition'].iloc[0]}   [{DATA_TYPE_TAG}]",
                     fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(f"Trust-Edge - Estimated Safe Time Remaining "
                 f"(cap = {ESTR_CAP_MIN:.0f} min means 'no crossing predicted')",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _stamp(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_analysis(all_df: pd.DataFrame, path: str) -> None:
    exps = sorted(all_df["experiment_id"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))

    # --- top: risk score of every experiment ---------------------------------
    ax = axes[0]
    palette = ["#2e7d32", "#f9a825", "#c62828", "#1565c0", "#6a1b9a", "#00897b"]
    for k, eid in enumerate(exps):
        d = all_df[all_df["experiment_id"] == eid]
        ax.plot(d["time_minutes"], d["risk_score"], lw=1.8,
                color=palette[k % len(palette)],
                label=f"Exp {eid} - {d['thermal_condition'].iloc[0]}")
    gates = [("MEDIUM", RISK_MEDIUM_SCORE, "MEDIUM score gate = 30"),
             ("HIGH", RISK_HIGH_SCORE, "HIGH score gate = 60"),
             ("CRITICAL", RISK_CRITICAL_MIN_SCORE,
              "CRITICAL companion gate = 50 (also needs ESTR <= 10 min)")]
    for lvl, val, txt in gates:
        ax.axhline(val, color=LEVEL_COLORS[lvl], ls="--", lw=0.9)
        ax.text(1, val + 1, txt, fontsize=7, color=LEVEL_COLORS[lvl])
    ax.set_ylabel("Risk score (0-100)")
    ax.set_xlabel("Time (minutes)")
    ax.set_title(f"Risk score - all experiments   [{DATA_TYPE_TAG}]", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # --- bottom: transparent breakdown for the worst-case experiment ---------
    worst = int(all_df.groupby("experiment_id")["risk_score"].max().idxmax())
    d = all_df[all_df["experiment_id"] == worst]
    t = d["time_minutes"].to_numpy()
    contrib = [
        100 * RISK_WEIGHTS["proximity"] * d["score_proximity"].to_numpy(),
        100 * RISK_WEIGHTS["temp_trend"] * d["score_temp_trend"].to_numpy(),
        100 * RISK_WEIGHTS["humidity_trend"] * d["score_humidity_trend"].to_numpy(),
        100 * RISK_WEIGHTS["lid"] * d["score_lid"].to_numpy(),
    ]
    ax = axes[1]
    ax.stackplot(t, *contrib,
                 labels=[f"Temperature proximity (w={RISK_WEIGHTS['proximity']})",
                         f"Temperature trend dT/dt (w={RISK_WEIGHTS['temp_trend']})",
                         f"Humidity trend dH/dt (w={RISK_WEIGHTS['humidity_trend']})",
                         f"Lid status (w={RISK_WEIGHTS['lid']})"],
                 colors=["#1565c0", "#ef6c00", "#00897b", "#8e24aa"], alpha=0.85)
    ax.set_ylabel("Contribution to risk score")
    ax.set_xlabel("Time (minutes)")
    ax.set_title(f"Explainable risk breakdown - Experiment {worst}   [{DATA_TYPE_TAG}]",
                 fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Trust-Edge - Transparent risk scoring", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _stamp(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_thermal_event(all_df: pd.DataFrame, path: str) -> None:
    """Full Sense -> Predict -> ESTR -> Decide -> Act timeline for the worst run."""
    worst = int(all_df.groupby("experiment_id")["risk_score"].max().idxmax())
    d = all_df[all_df["experiment_id"] == worst].reset_index(drop=True)
    t = d["time_minutes"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1.2, 1.0]})

    def mark(ax, level, color, slot):
        idx = d.index[d["risk_level"] == level]
        if len(idx):
            x = d["time_minutes"].iloc[idx[0]]
            ax.axvline(x, color=color, ls="--", lw=1.1)
            lo, hi = ax.get_ylim()
            y = hi - (hi - lo) * (0.05 + 0.11 * slot)
            ax.text(x + 0.8, y, f"first {level}  {x:.1f} min",
                    fontsize=7, color=color, va="top")

    # --- temperature ----------------------------------------------------------
    ax = axes[0]
    ax.plot(t, d["ground_truth_payload_temperature"], color="#1565c0", lw=2.0,
            label="Payload Tp")
    ax.plot(t, d["ambient_temperature"], color="#9e9e9e", lw=1.0, label="Ambient Ta")
    ax.axhline(SAFETY_THRESHOLD_C, color="#6a1b9a", ls="-.", lw=1.2,
               label=f"Threshold {SAFETY_THRESHOLD_C:.1f} C")
    opened = d[d["lid_status"] == "OPEN"]
    if len(opened):
        x0 = opened["time_minutes"].iloc[0]
        ax.axvspan(x0, t[-1], color="#ffcdd2", alpha=0.25)
        ax.text(x0 + 0.6, ax.get_ylim()[0] + 0.5, f"lid OPEN @ {x0:.0f} min", fontsize=7)
    for slot, lvl in enumerate(["MEDIUM", "HIGH", "CRITICAL"]):
        mark(ax, lvl, LEVEL_COLORS[lvl], slot)
    # actual threshold crossing in the simulation ground truth -> shows lead time
    cross = d.index[d["ground_truth_payload_temperature"] >= SAFETY_THRESHOLD_C]
    if len(cross):
        xc = d["time_minutes"].iloc[cross[0]]
        ax.axvline(xc, color="#000000", lw=1.4)
        ax.text(xc + 0.8, ax.get_ylim()[0] + 1.6,
                f"actual breach {xc:.1f} min", fontsize=7, color="#000000")
        _ = xc
        dep_idx = d.index[d["actuator_state"] == "DEPLOYED"]
        if len(dep_idx):
            xa = d["time_minutes"].iloc[dep_idx[0]]
            ax.annotate("", xy=(xc, SAFETY_THRESHOLD_C + 2.2),
                        xytext=(xa, SAFETY_THRESHOLD_C + 2.2),
                        arrowprops=dict(arrowstyle="<->", color="#c62828", lw=1.2))
            ax.text((xa + xc) / 2.0, SAFETY_THRESHOLD_C + 2.6,
                    f"acted {xc - xa:.1f} min before the breach",
                    fontsize=7.5, color="#c62828", ha="center")
    ax.set_ylabel("Temperature (C)")
    ax.set_title(f"Thermal event timeline - Experiment {worst}   [{DATA_TYPE_TAG}]", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # --- ESTR -----------------------------------------------------------------
    ax = axes[1]
    ax.plot(t, d["ESTR_minutes"], color="#2e7d32", lw=1.6, label="ESTR (thermal model)")
    ax.plot(t, d["ESTR_minutes_linear"], color="#c62828", lw=1.2, ls=":",
            label="ESTR (linear baseline)")
    ax.axhline(ESTR_CRITICAL_MIN, color="#6a1b9a", ls="-.", lw=1.0)
    ax.set_ylabel("ESTR (min)")
    ax.set_ylim(0, min(ESTR_CAP_MIN, 120))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # --- decision + actuation -------------------------------------------------
    ax = axes[2]
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    y = d["risk_level"].map(order).to_numpy()
    ax.step(t, y, where="post", color="#37474f", lw=1.6, label="Decision engine state")
    ax.set_yticks(list(order.values()))
    ax.set_yticklabels(list(order.keys()))
    ax.set_ylim(-0.4, 3.6)
    dep = d.index[d["actuator_state"] == "DEPLOYED"]
    if len(dep):
        x = d["time_minutes"].iloc[dep[0]]
        ax.axvline(x, color="#c62828", lw=1.6)
        ax.text(x + 0.6, 0.2,
                f"EMERGENCY ACTUATION\n(servo one-shot release)\n{x:.1f} min",
                fontsize=7, color="#c62828")
    ax.set_xlabel("Time (minutes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Trust-Edge - Sense / Predict / ESTR / Decide / Act",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _stamp(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ==============================================================================
# 7b. REAL-DATA MODE  (drop-in replacement for the simulated CSV)
# ==============================================================================

REQUIRED_REAL_COLUMNS = ["payload_temperature", "ambient_temperature", "humidity"]


def load_real_csv(path: str) -> pd.DataFrame:
    """
    Load a REAL recording (for example an ESP32 + DS18B20 + DHT11 log) and put
    it into exactly the same shape the simulator produces, so the identical
    pipeline can be run on it.

    Minimum required columns:
        payload_temperature, ambient_temperature, humidity
    Optional columns:
        timestamp | time_s | time_minutes, lid_status, experiment_id,
        thermal_condition
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_REAL_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: input file is missing required column(s): {missing}")

    # ---- time base -----------------------------------------------------------
    if "time_s" in df.columns:
        t_s = df["time_s"].astype(float).to_numpy()
    elif "time_minutes" in df.columns:
        t_s = df["time_minutes"].astype(float).to_numpy() * 60.0
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        t_s = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    else:
        t_s = np.arange(len(df), dtype=float) * SAMPLE_INTERVAL_S
        print("WARNING: no time column found, assuming a uniform "
              f"{SAMPLE_INTERVAL_S:.0f} s sampling interval.")

    out = pd.DataFrame({
        "time_s": t_s,
        "time_minutes": t_s / 60.0,
        "experiment_id": df["experiment_id"] if "experiment_id" in df.columns else 1,
        "thermal_condition": (df["thermal_condition"] if "thermal_condition" in df.columns
                              else "real_measurement"),
        "payload_temperature": df["payload_temperature"].astype(float),
        "ambient_temperature": df["ambient_temperature"].astype(float),
        "humidity": df["humidity"].astype(float),
        "lid_status": (df["lid_status"].astype(str).str.upper()
                       if "lid_status" in df.columns else "CLOSED"),
    })
    # For a real recording the measured payload temperature IS the reference the
    # prediction is scored against (post-hoc validation of what actually happened).
    out["ground_truth_payload_temperature"] = out["payload_temperature"]
    out["ground_truth_ambient_temperature"] = out["ambient_temperature"]
    out["ground_truth_tau_s"] = np.nan
    if "timestamp" in df.columns:
        out["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        out["timestamp"] = [SIM_START_TIME + timedelta(seconds=float(t)) for t in t_s]
    return out


def run_real_mode(path: str) -> None:
    global DATA_TYPE_TAG, SIM_BANNER
    DATA_TYPE_TAG = "MEASURED"
    SIM_BANNER = f"REAL RECORDED DATA - source file: {os.path.basename(path)}"

    raw = load_real_csv(path)
    frames = [process_experiment(g) for _, g in raw.groupby("experiment_id", sort=True)]
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[[c for c in COLUMN_ORDER if c in all_df.columns]]

    out_csv = os.path.join(OUTPUT_DIR, "trust_edge_real_data_processed.csv")
    all_df.to_csv(out_csv, index=False)
    metrics = [compare_models(f) for f in frames]

    plot_temperature_prediction(all_df, os.path.join(OUTPUT_DIR, "real_temperature_prediction.png"), metrics)
    plot_estr(all_df, os.path.join(OUTPUT_DIR, "real_estr_prediction.png"))
    plot_risk_analysis(all_df, os.path.join(OUTPUT_DIR, "real_risk_analysis.png"))
    plot_thermal_event(all_df, os.path.join(OUTPUT_DIR, "real_thermal_event.png"))
    print_summary(all_df, metrics)
    print(f"Processed real recording written to: {out_csv}")


# ==============================================================================
# 8. TERMINAL SUMMARY
# ==============================================================================

def print_summary(all_df: pd.DataFrame, metrics: list[dict]) -> None:
    line = "=" * 78
    print()
    print(line)
    print("TRUST-EDGE THERMAL SIMULATION - SUMMARY")
    print(line)
    print(f"  *** {SIM_BANNER} ***")
    print(line)

    n_exp = all_df["experiment_id"].nunique()
    print(f"Number of experiments        : {n_exp}")
    print(f"Total number of samples      : {len(all_df)}")
    print(f"Sampling interval            : {SAMPLE_INTERVAL_S:.0f} s")
    print(f"Safety threshold (prototype) : {SAFETY_THRESHOLD_C:.2f} C")
    print(f"Prediction horizon           : {PREDICTION_HORIZON_MIN:.0f} min")
    print(f"Master random seed           : {MASTER_SEED}")
    print(line)

    for eid, d in all_df.groupby("experiment_id"):
        crit = d[d["risk_level"] == "CRITICAL"]
        dep = d[d["actuator_state"] == "DEPLOYED"]
        print(f"EXPERIMENT {eid}  ({d['thermal_condition'].iloc[0]})")
        print(f"  samples                    : {len(d)}")
        print(f"  duration                   : {d['time_minutes'].max():.1f} min")
        print(f"  initial payload temperature: {d['payload_temperature'].iloc[0]:.3f} C")
        print(f"  maximum payload temperature: {d['payload_temperature'].max():.3f} C")
        print(f"  minimum payload temperature: {d['payload_temperature'].min():.3f} C")
        print(f"  ambient range              : {d['ambient_temperature'].min():.2f} "
              f"-> {d['ambient_temperature'].max():.2f} C")
        print(f"  minimum ESTR (thermal)     : {d['ESTR_minutes'].min():.2f} min")
        print(f"  minimum ESTR (linear)      : {d['ESTR_minutes_linear'].min():.2f} min")
        print(f"  maximum risk score         : {d['risk_score'].max():.2f} / 100")
        print(f"  highest risk level reached : "
              f"{max(d['risk_level'], key=lambda s: {'LOW':0,'MEDIUM':1,'HIGH':2,'CRITICAL':3}[s])}")
        if len(crit):
            print(f"  first CRITICAL at          : {crit['time_minutes'].iloc[0]:.2f} min")
        else:
            print("  first CRITICAL at          : never (no CRITICAL state in this run)")
        if len(dep):
            print(f"  emergency actuation at     : {dep['time_minutes'].iloc[0]:.2f} min "
                  f"(after {ACTUATION_PERSISTENCE_S:.0f} s persistence)")
        else:
            print("  emergency actuation at     : not triggered")
        print("-" * 78)

    print("MODEL COMPARISON  (thermal / RC model vs linear baseline)")
    print(f"{'Exp':>4} | {'RMSE RC':>9} | {'RMSE LIN':>9} | {'ESTR MAE RC':>12} | "
          f"{'ESTR MAE LIN':>13} | {'Lead time':>10} | {'n':>5}")
    print("-" * 90)
    for m in metrics:
        lead = "n/a" if math.isnan(m["lead_time_min"]) else f"{m['lead_time_min']:.1f} min"
        mae_rc = "n/a" if math.isnan(m["estr_mae_thermal_min"]) else f"{m['estr_mae_thermal_min']:.2f} min"
        mae_ln = "n/a" if math.isnan(m["estr_mae_linear_min"]) else f"{m['estr_mae_linear_min']:.2f} min"
        print(f"{m['experiment_id']:>4} | {m['rmse_thermal_c']:>7.3f} C | "
              f"{m['rmse_linear_c']:>7.3f} C | {mae_rc:>12} | {mae_ln:>13} | {lead:>10} | "
              f"{m['estr_samples']:>5}")
    print("-" * 90)
    print("RMSE  = error of the 10-minute-ahead temperature prediction vs ground truth.")
    print("ESTR MAE = mean absolute error of the remaining-safe-time estimate.")
    print("Lead time = how early HIGH/CRITICAL was raised before the true crossing.")
    print("n     = samples inside the fair ESTR evaluation window (disturbance active,")
    print("        true crossing still ahead, both estimators reporting a finite time).")
    print(line)
    if DATA_TYPE_TAG == "SIMULATED":
        print("REMINDER: these numbers characterise the SIMULATION only. They become")
        print("evidence only after the same pipeline is re-run on recorded ESP32 /")
        print("DS18B20 / DHT11 measurements.")
    else:
        print("These numbers were produced from a REAL recording. State clearly in any")
        print("report which sensors, enclosure and conditions produced the file.")
    print(line)


# ==============================================================================
# 9. MAIN
# ==============================================================================

COLUMN_ORDER = [
    "timestamp", "time_minutes", "experiment_id", "thermal_condition",
    "payload_temperature", "ambient_temperature", "humidity", "lid_status",
    "dT_dt", "dH_dt",
    "predicted_temperature", "predicted_temperature_linear",
    "risk_score", "risk_level",
    "ESTR_minutes", "ESTR_low_minutes", "ESTR_high_minutes",
    "ESTR_minutes_linear", "ESTR_minutes_ground_truth",
    "score_proximity", "score_temp_trend", "score_humidity_trend", "score_lid",
    "tau_estimated_s", "estimator_status", "actuator_state",
    "ground_truth_payload_temperature", "ground_truth_ambient_temperature",
    "ground_truth_tau_s", "data_type",
]


def run_simulation_mode() -> None:
    frames = []
    for cfg in EXPERIMENTS:
        raw = simulate_experiment(cfg)
        processed = process_experiment(raw)
        frames.append(processed)

    all_df = pd.concat(frames, ignore_index=True)

    all_df = all_df[COLUMN_ORDER]

    csv_path = os.path.join(OUTPUT_DIR, "trust_edge_simulation.csv")
    all_df.to_csv(csv_path, index=False)

    metrics = [compare_models(f) for f in frames]

    plot_temperature_prediction(all_df, os.path.join(OUTPUT_DIR, "temperature_prediction.png"),
                                metrics)
    plot_estr(all_df, os.path.join(OUTPUT_DIR, "estr_prediction.png"))
    plot_risk_analysis(all_df, os.path.join(OUTPUT_DIR, "risk_analysis.png"))
    plot_thermal_event(all_df, os.path.join(OUTPUT_DIR, "thermal_event.png"))

    print_summary(all_df, metrics)

    print("Files written:")
    for f in ["trust_edge_simulation.csv", "temperature_prediction.png",
              "estr_prediction.png", "risk_analysis.png", "thermal_event.png"]:
        print(f"  - {os.path.join(OUTPUT_DIR, f)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trust-Edge physics-informed cold-chain thermal pipeline.")
    parser.add_argument(
        "--input", metavar="CSV", default=None,
        help=("Run the SAME pipeline on a real recorded CSV (for example an ESP32 / "
              "DS18B20 / DHT11 log) instead of generating simulated data. Required "
              "columns: payload_temperature, ambient_temperature, humidity."))
    args = parser.parse_args()

    if args.input:
        run_real_mode(args.input)
    else:
        run_simulation_mode()


if __name__ == "__main__":
    main()
