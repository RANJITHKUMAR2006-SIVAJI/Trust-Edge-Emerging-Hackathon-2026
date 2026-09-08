# Trust-Edge — Physics-Informed Cold-Chain Thermal Simulation Dataset

**Module:** `Sense → Predict → Estimate Safe Time Remaining (ESTR) → Decide → Act → Log`
**Version:** baseline v1.0 (no machine learning, by design)

---

## ⚠️ 1. MANDATORY DATA-PROVENANCE NOTICE — READ FIRST

> **EVERY NUMBER IN `trust_edge_simulation.csv` AND EVERY POINT ON THE FOUR PNG
> FIGURES IS SIMULATED / MODEL-GENERATED DATA.**
>
> **It was produced by integrating a differential equation on a laptop.
> It is NOT a physical sensor measurement.
> It must NEVER be presented, published, submitted or defended as
> experimental evidence, test results, or field data.**

Safeguards built into the project so this cannot be misrepresented by accident:

| Safeguard | Where |
|---|---|
| `data_type` column set to `SIMULATED` on every single row | `trust_edge_simulation.csv` |
| `[SIMULATED]` tag in every subplot title | all 4 PNG files |
| Red footer banner "SIMULATED DATA – MODEL GENERATED, NOT PHYSICAL MEASUREMENT" | all 4 PNG files |
| Banner printed at the top and bottom of the terminal summary | `trust_edge_simulation.py` |
| Ground-truth columns explicitly named `ground_truth_*` | CSV |

The correct sentence to use anywhere this dataset appears:

> *"Figure X shows a **model-generated** thermal simulation used to develop and
> unit-test the Trust-Edge prediction pipeline. Physical validation on
> ESP32 / DS18B20 / DHT11 hardware is the next step."*

---

## 2. What this project actually is

Trust-Edge needs a prediction pipeline **before** the sensors are wired up, so
that on demo day the only unknown is the hardware — not the algorithm.

This project therefore does two separate things:

1. **A plant simulator** — generates physically plausible payload/ambient/humidity
   traces by integrating a first-order thermal ODE (this is the *fake sensor*).
2. **The Trust-Edge edge pipeline** — derivative estimation, online thermal-parameter
   identification, prediction, ESTR, transparent risk scoring, a decision engine
   and one-shot actuation logic (this is the *real deliverable*, and it is
   written so it can run unchanged on real data).

The second part is the part that will later be ported to ESP32 firmware and then
to the PYNQ-Z2. The first part gets thrown away the day real data exists.

**The script can already run the pipeline on real data:**

```bash
python trust_edge_simulation.py --input my_esp32_log.csv
```

---

## 3. Equations used

### 3.1 Plant / ground-truth model (first-order Newton-style thermal model)

```
dTp/dt = (Ta - Tp) / tau
```

| Symbol | Meaning | Unit |
|---|---|---|
| `Tp` | payload temperature | °C |
| `Ta` | ambient temperature | °C |
| `tau` | effective thermal time constant of the enclosure, `tau = Rth · Cth` | s |

Integrated with the exact zero-order-hold solution over each sampling step
(unconditionally stable, no Euler drift):

```
Tp(t + dt) = Ta + ( Tp(t) - Ta ) · exp( -dt / tau )
```

Humidity is relaxed towards a target with the same structure and its own time
constant `tau_H`.

### 3.2 Predictor A — thermal (RC) model, with **online** parameter identification

`tau` is **not** assumed known by the predictor. It is identified live from the
measured signals, exactly as firmware would:

```
tau_hat = ( Ta_measured - Tp_measured ) / (dTp/dt)_measured
```

and the forecast at horizon `H` is:

```
T_pred(t + H) = Ta + ( Tp - Ta ) · exp( -H / tau_hat )
```

`tau_hat` is smoothed with a single-pole filter (α = 0.25). When the candidate
value jumps by more than 2× — which is what physically happens when a lid opens
and the enclosure regime changes — the filter switches to fast adaptation
(α = 0.60).

### 3.3 Predictor B — linear baseline (the control this must be compared against)

```
T_pred(t + H) = Tp + (dT/dt) · H
```

### 3.4 Derivatives

`dT_dt` and `dH_dt` are **not** simple two-point differences (too noisy). They are
least-squares slopes over a trailing 10-minute window, then smoothed with an EMA
(α = 0.30), and reported per **minute**. This is directly implementable on an
ESP32 with a ring buffer.

### 3.5 ESTR — Estimated Safe Time Remaining

Solve for the time at which the predicted trajectory reaches the safety threshold
`T_thr`:

**Thermal model (closed form):**

```
Ta + (Tp - Ta)·exp(-t/tau)  =  T_thr
⇒  ESTR = tau_hat · ln( (Ta - Tp) / (Ta - T_thr) )
```

**Linear baseline:**

```
ESTR_linear = ( T_thr - Tp ) / (dT/dt)
```

Special cases, both handled explicitly in the code:

| Condition | ESTR reported |
|---|---|
| `Tp ≥ T_thr` (already breached) | `0` |
| `Ta ≤ T_thr` (ambient can never push the payload past the limit) | cap (240 min) |
| `dT/dt ≤ 0.015 °C/min` (inside the slope noise floor) | cap (240 min) |
| estimator ring buffer not yet full (first 10 min) | cap, `estimator_status = WARMUP` |

**ESTR is reported as a range, never as a single exact number** — as required by
the Trust-Edge master specification. `ESTR_low_minutes` / `ESTR_high_minutes`
are ESTR ± the trailing 10-minute standard deviation of the estimate itself
(floor ±1 min), so the operator sees e.g. **"ESTR ≈ 21–27 min"**, not "ESTR = 23.4 min".

---

## 4. Assumptions (state these openly — they are the honest limits of the model)

1. **Lumped capacitance.** The payload is one uniform thermal mass at a single
   temperature. Real payloads have internal gradients.
2. **Single dominant heat path.** All heat transfer is collapsed into one `tau`.
   Radiation, forced convection, door-gap airflow and coolant phase change are
   *not* modelled separately.
3. **No phase-change / coolant term.** A real cold box with an ice pack or PCM
   holds a near-constant temperature until the coolant is exhausted, then rises.
   That plateau is **not** in this model. The chosen large `tau` values imitate
   the *slowness* of such a box, not its physics.
4. **Ambient is an independent input**, unaffected by the payload.
5. **Lid state is a known binary input** that switches `tau` between two values.
6. **Sensor errors are zero-mean and independent.** Real DS18B20 self-heating,
   thermal lag of the probe sheath, and DHT11 hysteresis are not modelled.
7. **Humidity is correlated with lid state by construction** in the simulator.
   That is an *assumption inside the generator*, therefore it can never be used
   as evidence that humidity detects a seal breach (see §8).
8. The safety threshold is a **prototype parameter**, not a product limit (§5).

---

## 5. Parameter values

### 5.1 Safety and decision parameters

| Parameter | Value | Note |
|---|---|---|
| `SAFETY_THRESHOLD_C` | **8.0 °C** | **PROTOTYPE DEMONSTRATION PARAMETER ONLY.** In a real deployment this must be replaced by the validated, product-specific storage limit from the manufacturer's storage label / pharmacopoeia (e.g. the upper bound of a 2–8 °C vaccine window). Change one line to change it. |
| `NOMINAL_SETPOINT_C` | 5.0 °C | baseline of the risk proximity term |
| `PREDICTION_HORIZON_MIN` | 10 min | how far ahead the predictors forecast |
| `ESTR_CAP_MIN` | 240 min | reporting horizon; the cap means "no crossing predicted" |
| `ACTUATION_PERSISTENCE_S` | 60 s | CRITICAL must persist this long before the actuator fires |

### 5.2 Signal-processing parameters

| Parameter | Value | Reason |
|---|---|---|
| `SAMPLE_INTERVAL_S` | 10 s | realistic logger rate; DS18B20 12-bit conversion is ~750 ms |
| `DERIV_WINDOW_S` | 600 s | LSQ slope window |
| `SLOPE_EMA_ALPHA` | 0.30 | extra slope smoothing |
| `MIN_SIGNIFICANT_TREND_C_PER_MIN` | 0.015 | noise floor gate. With σ = 0.05 °C noise and 60 samples the slope standard error is ≈ 0.002 °C/min, so this gate is ≈ 7σ — the pipeline refuses to extrapolate a crossing time out of noise |
| `TAU_EMA_ALPHA` / `_FAST` | 0.25 / 0.60 | normal / regime-change adaptation |
| `TAU_MIN_S` … `TAU_MAX_S` | 300 … 400 000 s | sanity clip on the identified `tau` |

### 5.3 Simulated sensor imperfection

| Sensor | Noise (1σ) | Resolution |
|---|---|---|
| DS18B20 (payload & ambient) | 0.05 °C | 0.0625 °C (12-bit) |
| DHT11 (humidity) | 0.4 %RH | 1 %RH |

Reproducibility: `MASTER_SEED = 20260911`, per-experiment seeds
`MASTER_SEED + experiment_id`, using `numpy.random.default_rng`. **Running the
script on any machine reproduces the CSV bit-for-bit.**

### 5.4 Experiment definitions

| # | Condition | Lid | `tau` closed → open | Ambient | Intended outcome |
|---|---|---|---|---|---|
| 1 | `normal_stable_lid_closed` | CLOSED throughout | 144 000 s (~40 h) | 25.0 → 25.6 °C | payload drifts ~1 °C, risk stays **LOW** |
| 2 | `moderate_disturbance_lid_opened` | opens at 20 min (ajar) | 144 000 → 54 000 s (~15 h) | 25.0 → 28.5 °C | payload rises toward but does **not** cross 8 °C, risk **LOW → MEDIUM → HIGH** |
| 3 | `severe_disturbance_lid_open` | opens at 15 min (fully) | 144 000 → 14 400 s (~4 h) | 25.0 → 34.0 °C | payload crosses 8 °C, risk reaches **CRITICAL**, actuator fires |
| 4 | `refrigeration_failure_low_ambient` | opens at 10 min | 144 000 → 7 200 s (~2 h) | 12.0 °C (failed chiller) | **diagnostic case:** ambient is only 4 °C above the limit, so the exponential curves hard and the two predictors visibly disagree |

Experiment 4 exists specifically to answer the question *"when does the physics
model actually beat the straight line?"* — see §9.

---

## 6. Risk scoring (transparent and fully auditable)

The risk score is a plain weighted sum of four normalised sub-scores. No hidden
model, no black box, no learned weights:

```
risk_score = 100 × [ 0.40·proximity + 0.30·temp_trend + 0.10·humidity_trend + 0.20·lid ]
```

| Sub-score | Definition | Saturates at |
|---|---|---|
| `score_proximity` | `(Tp − 5 °C) / (8 °C − 5 °C)`, clipped to [0, 1] | payload at the threshold |
| `score_temp_trend` | `(dT/dt) / 0.10 °C·min⁻¹`, clipped to [0, 1] | 0.10 °C/min |
| `score_humidity_trend` | `(dH/dt) / 0.50 %RH·min⁻¹`, clipped to [0, 1] | 0.50 %RH/min |
| `score_lid` | 1 if lid OPEN, else 0 | — |

**All four sub-scores are written into the CSV**, so any reviewer can recompute
the score by hand from the row and confirm the arithmetic.

### Decision engine

| Level | Condition |
|---|---|
| **CRITICAL** | `Tp ≥ threshold` **OR** (`ESTR ≤ 10 min` **AND** `risk_score ≥ 50`) |
| **HIGH** | `ESTR ≤ 30 min` **OR** `risk_score ≥ 60` |
| **MEDIUM** | `ESTR ≤ 60 min` **OR** `risk_score ≥ 30` |
| **LOW** | otherwise |

Two deliberate safety properties:

- **A high score alone can never fire the actuator.** CRITICAL requires an actual
  or predicted *threshold breach*. This prevents a single loud indicator (an open
  lid, a humidity spike) from deploying the one-shot mechanism.
- **Persistence check.** CRITICAL must hold continuously for 60 s (6 consecutive
  samples) before `actuator_state` becomes `DEPLOYED`. Deployment is irreversible
  in the real product, so a single noisy sample must never cause it.

---

## 7. Output files

| File | Content |
|---|---|
| `trust_edge_simulation.py` | the generator **and** the full pipeline (single file, ~1 100 lines, heavily commented) |
| `trust_edge_simulation.csv` | 2 884 rows × 30 columns, 4 experiments |
| `temperature_prediction.png` | payload/ambient traces + both 10-min-ahead predictions + per-experiment RMSE box |
| `estr_prediction.png` | ESTR (thermal) with uncertainty band, ESTR (linear), and the true remaining time |
| `risk_analysis.png` | risk score for all experiments + the explainable stacked sub-score breakdown |
| `thermal_event.png` | the full event timeline: lid opening → MEDIUM → HIGH → CRITICAL → actuation → actual breach |
| `README.md` | this document |

### CSV columns

`timestamp`, `time_minutes`, `experiment_id`, `thermal_condition`,
`payload_temperature`, `ambient_temperature`, `humidity`, `lid_status`,
`dT_dt`, `dH_dt`, `predicted_temperature`, `predicted_temperature_linear`,
`risk_score`, `risk_level`, `ESTR_minutes`, `ESTR_low_minutes`,
`ESTR_high_minutes`, `ESTR_minutes_linear`, `ESTR_minutes_ground_truth`,
`score_proximity`, `score_temp_trend`, `score_humidity_trend`, `score_lid`,
`tau_estimated_s`, `estimator_status`, `actuator_state`,
`ground_truth_payload_temperature`, `ground_truth_ambient_temperature`,
`ground_truth_tau_s`, `data_type`

- `predicted_temperature` = thermal (RC) model forecast for `t + 10 min`.
- `ESTR_minutes` = thermal-model ESTR (this is *the* ESTR, per specification).
- `*_ground_truth` / `ground_truth_*` columns exist **only** for scoring the
  estimators. They are never read by the pipeline itself.

---

## 8. Limitations (say these out loud before anyone asks)

1. **It is simulated.** Nothing here validates a physical enclosure, a physical
   sensor, or a physical actuator.
2. **The plant and the predictor share a model family.** The data is generated by
   a first-order system and one of the predictors *is* a first-order model. This
   flatters the RC model. Real enclosures are not perfectly first-order, so the
   real-data errors will be larger. This is the single biggest caveat.
3. **No coolant/PCM plateau, no thermal mass gradients, no radiation.**
4. **Humidity and lid status prove nothing about seal breach here.** The
   correlation between them was *put in by hand* in the generator. Per the
   Trust-Edge specification they remain *corroborating* features only, and both
   carry small weights (0.10 and 0.20) precisely for that reason.
5. **VOC / pressure / GPS are not included** in this baseline.
6. **The 8 °C threshold is arbitrary** and is a prototype parameter (§5.1).
7. **`tau` identification needs an active trend.** During the ~10 min after a lid
   transition the trailing slope window still holds pre-event samples, so ESTR is
   transiently optimistic. That settling interval is visible in
   `estr_prediction.png` and is excluded from the metric table (and this
   exclusion is stated in the terminal output — it is disclosed, not hidden).
8. **No security layer yet.** HMAC-SHA256 chaining is a later priority
   (Priority 8 in the master plan); this dataset is plain CSV.
9. **No machine learning**, deliberately. This is the physics baseline that any
   future ML model must be proven to beat.

---

## 9. Results of this run (simulated)

### Behaviour

| Exp | Max payload | Highest risk | Min ESTR | First CRITICAL | Actuation | True breach |
|---|---|---|---|---|---|---|
| 1 | 6.06 °C | LOW (18.9) | 240 (cap) | never | none | none |
| 2 | 7.50 °C | HIGH (60.6) | 21.1 min | never | none | none |
| 3 | 13.88 °C | CRITICAL (87.9) | 0 min | 39.8 min | 40.7 min | 49.7 min |
| 4 | 9.25 °C | CRITICAL (70.8) | 0 min | 64.5 min | 66.7 min | 76.7 min |

Experiments 1–3 behave exactly as specified. In experiments 3 and 4 the emergency
mechanism was commanded **9.0 min and 10.0 min before** the payload actually
crossed the limit — which is the entire point of the Trust-Edge architecture.

### Thermal (RC) model vs linear baseline

| Exp | 10-min RMSE (RC) | 10-min RMSE (linear) | ESTR MAE (RC) | ESTR MAE (linear) | Lead time |
|---|---|---|---|---|---|
| 1 | 0.072 °C | **0.057 °C** | – | – | – |
| 2 | 0.083 °C | **0.070 °C** | – | – | – |
| 3 | 0.221 °C | **0.199 °C** | 0.74 min | **0.53 min** | 27.2 min |
| 4 | **0.132 °C** | 0.126 °C | **1.73 min** | 5.20 min | 34.5 min |

**Read this honestly — it is the most valuable result in the whole project:**

> In experiments 1–3 the linear baseline is *as good as or slightly better than*
> the thermal model. The thermal model only wins decisively in experiment 4
> (ESTR error 1.73 min vs 5.20 min, i.e. **3× better**).

And there is a clean physical reason. Over a short span the exponential is nearly
straight; the curvature only matters when the payload consumes a large fraction of
the driving gradient `(Ta − Tp)`. In experiments 1–3 the payload travels 3 °C out of
a 20 °C gradient (~15 %) — nearly linear. In experiment 4 it travels 3 °C out of a
7 °C gradient (~43 %) — strongly curved, and the linear extrapolation under-estimates
the remaining time by ~5 min.

**Therefore the correct claim for Trust-Edge is:**

> *"The physics-informed model is not universally better. It is decisively better
> in the regime where the payload approaches ambient — which is exactly the
> refrigeration-failure and long-horizon regime that matters most for
> cold-chain safety. Whether this holds on the real enclosure is an open
> question that the ESP32 experiments will answer."*

That statement satisfies the master specification's rule: *do not claim the RC
model is better until data demonstrates it.*

---

## 10. How to run

Requirements: Python 3.9+, `numpy`, `pandas`, `matplotlib`.

```bash
pip install numpy pandas matplotlib
python trust_edge_simulation.py
```

Runtime is a few seconds on a normal laptop. It regenerates the CSV, all four
PNGs and the terminal summary. No internet, no GPU, no display needed
(matplotlib runs on the `Agg` backend).

To change the safety threshold, sampling rate, weights or experiments, edit the
clearly-marked configuration block at the top of the file — every constant lives
there, none are buried in the code.

---

## 11. How this dataset gets replaced by real ESP32 data

This is the whole point of the exercise. The pipeline is already written to
accept a real log.

### Step 1 — log this exact header from the ESP32

```
timestamp,payload_temperature,ambient_temperature,humidity,lid_status
2026-09-11T09:00:00,5.06,24.94,45,CLOSED
2026-09-11T09:00:10,5.06,25.00,45,CLOSED
```

- `payload_temperature` ← DS18B20 #1 (inside the box, near the payload)
- `ambient_temperature` ← DS18B20 #2 (outside the box)
- `humidity` ← DHT11
- `lid_status` ← reed switch / magnetic sensor, or typed by hand during the run
- Required columns are only the three temperatures/humidity; `timestamp`,
  `lid_status`, `experiment_id` and `thermal_condition` are optional.

### Step 2 — run the identical pipeline on it

```bash
python trust_edge_simulation.py --input esp32_run1.csv
```

This writes `trust_edge_real_data_processed.csv` plus `real_*.png`, with
`data_type = MEASURED` and the `[MEASURED]` tag replacing `[SIMULATED]` on every
figure. **Not one line of the prediction, ESTR, risk or decision code changes.**

### Step 3 — the swap that makes the numbers real

| Now (simulated) | After the ESP32 runs |
|---|---|
| `tau` chosen in the config | `tau` identified from the real enclosure and cross-checked against `tau_hat` |
| noise model assumed (0.05 °C) | noise measured from a steady-state recording |
| ground truth = the ODE | ground truth = what the DS18B20 actually recorded later |
| threshold = 8 °C prototype value | threshold = validated product storage limit |
| RMSE / ESTR MAE = model self-consistency | RMSE / ESTR MAE = **genuine experimental evidence** |

### Step 4 — the train/validate discipline from the master specification

Fit / tune on **experiment 1**, then predict **experiment 2** without re-tuning,
and report the error on experiment 2. Never fit and score on the same run.

### Step 5 — then, and only then, migrate to PYNQ-Z2

The functions map one-to-one onto the migration plan: `rolling_slope_per_min`
(Migration 2) → the RC prediction and `tau_hat` identification (Migration 3) →
ESTR (Migration 4) → decision engine (Migration 5).

---

## 12. Using this in the hackathon presentation — allowed and forbidden claims

### ✅ You MAY say

- "We built and unit-tested the full Trust-Edge decision pipeline against a
  physics-based thermal simulation before touching hardware."
- "The pipeline is **deterministic and reproducible** — same seed, same CSV,
  bit-for-bit, on any machine."
- "The risk score is **fully explainable**: four weighted terms, all four written
  into every log row, recomputable by hand."
- "In simulation the system commanded emergency protection **9–10 minutes before**
  the payload crossed the limit, and did **not** trigger in the normal and
  moderate runs — so it is not simply an alarm that fires on everything."
- "We deliberately included a case where the **linear baseline wins**, because our
  engineering rule is to prove the physics model rather than assume it."
- "The same script runs on real sensor data with a single command-line flag —
  the algorithm is already hardware-ready."

### ❌ You MUST NOT say

- ~~"We tested it and got 0.13 °C accuracy."~~ → say *"in simulation"*, every time.
- ~~"Our system detects seal breach."~~ → humidity/lid correlation was assumed by
  the generator; nothing here detects a breach.
- ~~"ESTR is accurate to under 2 minutes."~~ → that is the model scoring itself.
- ~~"Validated", "tested", "measured", "experimental results", "field data"~~ for
  anything in these files. The word is **simulated**.
- ~~"The physics model beats the linear model."~~ → it does so in one of four
  simulated regimes. Say exactly that.
- Do not present the PNGs with the red banner cropped off.

### Suggested slide wording

> **Slide title:** Predictive pipeline — simulated validation
> **Sub-caption (keep it on the slide):** *Model-generated data. Physical
> validation on ESP32 + DS18B20 + DHT11 is the next milestone.*
> **Spoken line:** *"This is our algorithm running against a thermal model, not
> against a sensor. It exists so that on hardware day the only variable is the
> hardware. Here is exactly what we still have to prove."*

Judges reward that sentence. A team that clearly separates *demonstrated*,
*validated*, *designed* and *future work* is trusted on everything else it claims.

---

## 13. What this baseline still owes the project

Priorities carried over from the Trust-Edge master specification:

1. Real thermal experiments on the insulated box → identify the real `tau`.
2. Train/validate split across separate physical runs.
3. Backup-reserve sizing from a measured SG90 stall-current profile.
4. HMAC-SHA256 chained event records (the log columns are already defined here).
5. PCM cartridge thermal characterisation.
6. ESP32 → PYNQ-Z2 migration of these exact functions.
