# Heizungsregelung mit MPC & Adaptiver KI

## Systemübersicht

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Radiator Control System                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐       │
│  │  Experiment    │     │  Adaptive      │     │  MPC           │       │
│  │  Runner        │────▶│  Model         │────▶│  Controller    │       │
│  │                │     │  (Online RLS)  │     │                │       │
│  └────────────────┘     └────────────────┘     └────────────────┘       │
│                                │                       │                 │
│                                ▼                       ▼                 │
│                         ┌────────────────┐     ┌────────────────┐       │
│                         │  Neural Net    │     │  Reinforcement │       │
│                         │  (Optional)    │     │  Learning (RL) │       │
│                         └────────────────┘     └────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                   Home Assistant Interface                    │       │
│  │  • Echte Temperatur lesen (Sensor)                           │       │
│  │  • Fake-Temperatur schreiben (Thermostat manipulieren)       │       │
│  │  • Fensterstatus lesen                                       │       │
│  │  • Außentemperatur lesen                                     │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                │                                         │
│                                ▼                                         │
│                    ┌────────────────────┐                               │
│                    │  Silvercrest       │                               │
│                    │  Zigbee Thermostat │                               │
│                    │  (via Zigbee2MQTT) │                               │
│                    └────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Kernidee: Temperatur-Offset-Manipulation

Da wir das Thermostat nicht direkt steuern können, "lügen" wir es an:
- **Soll-Temp am Thermostat**: z.B. 21°C (fest eingestellt)
- **Fake-Temp**: Wenn wir mehr heizen wollen → niedrigere Temp melden
- **Effekt**: Thermostat denkt es ist kalt → heizt mehr

```
fake_temp = real_temp - offset
offset > 0  → mehr heizen
offset < 0  → weniger heizen
offset ∈ [-5, +5] °C (sinnvoller Bereich)
```

## Regelungsansätze

### 1. MPC (Model Predictive Control)
- Optimiert über einen Zeithorizont (z.B. 2 Stunden)
- Berücksichtigt Vorhersagen (Außentemp, Fensteröffnung geplant?)
- Braucht ein Systemmodell

### 2. Adaptive Modellierung (RLS - Recursive Least Squares)
- Modellparameter werden laufend angepasst
- Reagiert auf Änderungen (Jahreszeit, Möbel umgestellt, etc.)
- Kombinierbar mit MPC

### 3. KI/RL (Reinforcement Learning)
- Agent lernt optimale Strategie durch Trial & Error
- Kann komplexe Zusammenhänge lernen
- Braucht viele Daten / Simulationsumgebung zum Vortraining
- **Empfehlung**: Als zweite Phase nach MPC-Baseline

## Systemmodell (PT1 mit Totzeit)

Vereinfachtes thermisches Modell:
```
dT_room/dt = (1/τ) * (K_heater * Q_heater + K_outside * T_outside - T_room)

Wobei:
- τ: Zeitkonstante des Raums (~1-4 Stunden)
- K_heater: Verstärkung Heizung
- K_outside: Kopplung Außentemperatur
- Q_heater: Heizleistung (0-1, geschätzt aus Thermostat-Verhalten)
```

Mit Fensteröffnung:
```
dT_room/dt = ... + K_window * window_open * (T_outside - T_room)
```

## Implementierungsschritte

### Schritt 1: Projektstruktur & Konfiguration
- [x] KONZEPT.md (diese Datei)
- [x] config.yaml (Konfiguration)
- [x] requirements.txt (Dependencies)

### Schritt 2: Home Assistant Interface
- [x] ha_client.py - REST API / WebSocket Client
- [x] Lesen: Temperatur, Fensterstatus, Außentemp
- [x] Schreiben: Fake-Temperatur an Thermostat

### Schritt 3: Datenbank & Logging
- [x] database.py - SQLite für Messdaten
- [x] Speichern aller Sensordaten + Stellgrößen

### Schritt 4: Experiment Runner
- [x] experiments.py - Sprungantwort, PRBS
- [x] Automatische Systemidentifikation

### Schritt 5: Adaptives Modell
- [x] model.py - RLS-basierte Parameteranpassung
- [x] Thermisches Raummodell

### Schritt 6: MPC Controller
- [x] mpc_controller.py - Model Predictive Control
- [x] Optimierung mit CVXPY (OSQP Solver)

### Schritt 7: RL Controller (Optional/Später)
- [ ] rl_controller.py - Reinforcement Learning
- [ ] Gym Environment für Training

### Schritt 8: Main Loop & Docker
- [x] main.py - Hauptschleife
- [x] Dockerfile & docker-compose.yml

## Dateistruktur

```
radiator_control/
├── KONZEPT.md
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── __init__.py
│   ├── ha_client.py      # Home Assistant Kommunikation
│   ├── database.py       # Datenbank & Logging
│   ├── experiments.py    # Systemidentifikation
│   ├── model.py          # Adaptives thermisches Modell
│   ├── mpc_controller.py # MPC Regler
│   ├── rl_controller.py  # RL Regler (später)
│   └── utils.py          # Hilfsfunktionen
├── main.py               # Hauptprogramm
└── data/
    └── measurements.db   # SQLite Datenbank
```

## Konfigurationsparameter

```yaml
# Home Assistant
ha_url: "http://homeassistant.local:8123"
ha_token: "YOUR_LONG_LIVED_TOKEN"

# Entities
entities:
  thermostat: "climate.silvercrest_thermostat"
  temp_sensor: "sensor.room_temperature"  # Externer Sensor (genauer)
  window_sensor: "binary_sensor.window"
  outside_temp: "sensor.outside_temperature"

# Regelung
control:
  target_temp: 21.0
  temp_tolerance: 0.3
  sample_time: 60  # Sekunden
  offset_min: -5.0
  offset_max: 5.0

# MPC
mpc:
  horizon: 120  # Minuten
  weight_comfort: 1.0
  weight_energy: 0.1
```

## Nächster Schritt

Ich erstelle jetzt die Dateien in folgender Reihenfolge:
1. config.yaml
2. requirements.txt
3. src/ha_client.py
4. src/database.py
5. src/model.py
6. src/mpc_controller.py
7. src/experiments.py
8. main.py
9. Docker-Dateien
