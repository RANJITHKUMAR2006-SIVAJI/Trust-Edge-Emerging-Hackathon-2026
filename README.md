Trust-Edge
Predictive Cold-Chain Monitoring and Emergency Preservation
> \*\*Predict → Decide Locally → Protect → Log\*\*
Trust-Edge is an edge-oriented cold-chain monitoring and protection
proof of concept designed to identify developing thermal risk before a
temperature-sensitive payload crosses its configured safety threshold.
The system combines real-time sensing, physics-informed thermal
prediction, Estimated Safe Time Remaining (ESTR), transparent risk
classification, local decision-making, emergency actuation, and event
logging.
---
1. Problem
Conventional cold-chain monitoring is primarily reactive: an alarm is
generated after a temperature limit has already been crossed.
For vaccines and temperature-sensitive pharmaceutical shipments, earlier
warning can provide more time for corrective action.
Trust-Edge therefore focuses on the question:
> \*\*How much safe time remains based on the current and predicted
> thermal behavior?\*\*
---
2. System Concept
``` text
SENSE → PREDICT → ESTR → DECIDE → ACT → LOG
```
SENSE
Collect: - Payload temperature --- DS18B20 - Ambient temperature ---
DS18B20 - Humidity --- DHT11 - Lid status --- Reed switch
PREDICT
Evaluate: - Temperature trend (`dT/dt`) - Physics-informed thermal RC
model - Linear baseline - 10-minute-ahead temperature prediction
ESTR
Estimate: - Remaining safe time - Predicted safety-threshold crossing -
ESTR uncertainty range
DECIDE
Use a transparent risk engine: - LOW - MEDIUM - HIGH - CRITICAL
Decision inputs include temperature proximity, temperature trend,
humidity trend, and lid status.
ACT
When a configured critical condition persists: - Local buzzer alert -
SG90 one-shot emergency release - Passive emergency thermal protection -
Event logging
---
3. Current Prototype
The current prototype uses an ESP32 as the sensor and control
platform.
Component               Function
---
ESP32                   Controller and sensor interface
DS18B20                 Payload temperature
DS18B20                 Ambient temperature in dual-sensor configuration
DHT11                   Humidity and temperature
Reed switch             Lid-status detection
SG90 servo              One-shot emergency release
Buzzer                  Local alert
Insulated thermal box   Cold-chain prototype enclosure
The exact configuration may change during validation.
---
4. Thermal Prediction
Trust-Edge currently evaluates two approaches in parallel.
Physics-informed RC model
A simplified thermal model is:
``` text
dTp/dt = (Ta - Tp) / τ
```
where `Tp` is payload temperature, `Ta` is ambient temperature, and `τ`
is the effective thermal time constant.
Linear baseline
The recent temperature rate of change is used as a baseline:
``` text
dT/dt
```
Both approaches are compared because prediction performance can depend
on the type of thermal disturbance.
---
5. ESTR --- Estimated Safe Time Remaining
ESTR estimates the time remaining before the predicted payload
temperature reaches the configured safety threshold.
``` text
Current temperature
        ↓
Thermal prediction
        ↓
Predicted threshold crossing
        ↓
Estimated Safe Time Remaining
```
ESTR is a decision-support estimate, not a guaranteed operational
lifetime. Its accuracy requires physical validation under representative
thermal conditions.
---
6. Emergency Protection
The emergency-response concept is:
``` text
Critical condition
      ↓
Persistence / confirmation check
      ↓
One-shot SG90 release
      ↓
Mechanical deployment
      ↓
Passive thermal protection
```
The objective is for the actuator to deploy the protection mechanism
rather than continuously consume power to maintain it.
PCM performance, mechanical reliability, backup-power performance, and
protection duration remain validation items.
---
7. Local-First Architecture
Critical functions are intended to operate locally without continuous
cloud connectivity.
``` text
Sensors
   ↓
ESP32 / Edge Controller
   ├── Prediction
   ├── ESTR
   ├── Risk Decision
   ├── Emergency Actuation
   └── Event Logging
```
Remote monitoring or cloud synchronization can be added as a
supplemental function; it is not intended to be required for the
critical local decision path.
---
8. Validation
Development uses both synthetic and recorded sensor data.
Current validation includes: - Four synthetic thermal scenarios - RC
model versus linear baseline - Predicted temperature behavior -
Safety-threshold crossing - ESTR estimation - Risk-state logic
A representative future dataset can contain:
``` text
timestamp
experiment\_id
thermal\_condition
payload\_temperature
ambient\_temperature
humidity
lid\_status
dT\_dt
predicted\_temperature
prediction\_error
risk\_score
risk\_level
ESTR\_minutes
decision
```
Important: synthetic data is model-generated validation data and
must not be presented as real sensor measurements.
Physical ESP32 validation is being developed to measure sensor noise,
prediction accuracy, detection latency, ESTR error, false
positives/negatives, actuator reliability, and thermal protection
duration.
---
9. Development Roadmap
Completed / implemented
Synthetic sensor-acquisition pipeline
Thermal scenario generation
Physics-informed RC model
Linear prediction baseline
ESTR estimator
Four-state risk engine
Simulation-based model comparison
Current
ESP32 sensor integration
Real sensor data acquisition
Integrated prediction and ESTR demonstration
Emergency actuator integration
Controlled thermal experiments
Planned
PCM protection validation
Independent emergency-power validation
Pressure and VOC sensing as corroborating features
Time-synchronized sensor fusion
HMAC-SHA256 event authentication
ESP32 ↔ PYNQ-Z2 integration
FPGA implementation/acceleration
Controlled pilot validation
Production qualification assessment
---
10. Repository Structure
``` text
trust-edge/
├── README.md
├── firmware/
│   ├── esp32/
│   └── pynq/
├── models/
│   ├── thermal\_rc/
│   ├── linear\_baseline/
│   ├── estr/
│   └── risk\_engine/
├── data/
│   ├── synthetic/
│   ├── recorded/
│   └── processed/
├── experiments/
│   ├── thermal\_scenarios/
│   └── validation/
├── hardware/
│   ├── wiring/
│   ├── schematics/
│   └── prototype\_photos/
├── dashboard/
└── docs/
```
---
11. Getting Started
Hardware
Start by testing each component independently:
ESP32
DS18B20
DHT11
Reed switch
Buzzer
SG90 servo
Then integrate:
``` text
Sensors
  ↓
ESP32
  ↓
Recorded CSV
  ↓
Prediction
  ↓
ESTR
  ↓
Risk decision
  ↓
Alert / actuator
  ↓
Event log
```
Software
The model-development environment uses Python 3.x. Project-specific
dependencies should be documented in a `requirements.txt` or equivalent
environment file as the repository develops.
---
12. Hardware Safety
Do not power an SG90 from the ESP32 3.3 V output.
Use an appropriate servo supply with a common ground.
Use the required DS18B20 pull-up resistor.
Verify sensor pinouts before powering the circuit.
Keep the mechanical actuator safe during testing.
Do not use this prototype for actual vaccine or pharmaceutical
storage.
---
13. Current Limitations
Trust-Edge is currently a proof-of-concept / integration-stage
system, not a qualified production cold-chain device.
Known limitations: - Physical thermal validation is ongoing. - ESTR
accuracy depends on calibration and thermal conditions. - Sensor
response characteristics affect prediction. - PCM protection has not
been fully validated. - Backup-power performance requires dedicated
testing. - Pressure/VOC fusion is not yet demonstrated. - HMAC-based
event integrity is not yet demonstrated. - FPGA integration is a later
development stage. - Regulatory and industry qualification have not been
completed.
---
14. Innovation
The project integrates four functions into one local protection loop:
``` text
PREDICT
   ↓
DECIDE LOCALLY
   ↓
PROTECT
   ↓
LOG
```
The objective is to move beyond simple temperature alarms toward
predictive, locally autonomous cold-chain protection with measurable
evidence of system events.
---
15. Project Status
Area                       Status
---
Thermal prediction         Simulation validated
Linear baseline            Implemented
ESTR                       Implemented / simulation-tested
Risk engine                Implemented
ESP32 hardware             Integration stage
Real sensor validation     In progress
Emergency actuation        Integration/testing stage
PCM protection             Validation pending
Backup power               Validation pending
VOC/pressure fusion        Planned
HMAC event integrity       Planned
PYNQ-Z2 / FPGA             Planned
Production qualification   Not completed
---
16. Hackathon Information
Project: Trust-Edge --- Predictive Cold-Chain Protection  
Institution: M. Kumarasamy College of Engineering  
Event: Emerging Technologies Hackathon 2026  
Application: ETH-07749317  
Technology Vertical: Services --- Health Tech
---
17. Disclaimer
Trust-Edge is an experimental engineering proof of concept. Simulation
results and prototype demonstrations do not constitute pharmaceutical,
vaccine, medical-device, regulatory, or production qualification. Any
real-world deployment involving temperature-sensitive healthcare
products requires appropriate calibration, validation, qualification,
cybersecurity assessment, and applicable regulatory review.
---
18. Contact
For collaboration, technical discussion, or demonstration inquiries,
please use the contact information associated with this repository.
---
Trust-Edge --- Predict before damage occurs. Decide locally. Protect
the payload. Preserve the evidence.
