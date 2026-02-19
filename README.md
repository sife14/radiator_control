# 🌡️ Radiator Control

**Intelligente Heizungssteuerung für Home Assistant mit Model Predictive Control (MPC) und adaptiver Modellierung.**

## 🚀 Installation

### Option A: HACS (empfohlen)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=sife14&repository=radiator_control&category=integration)

1. **HACS öffnen** → Integrationen → ⋮ (oben rechts) → **Benutzerdefinierte Repositories**
2. URL: `https://github.com/sife14/radiator_control` | Kategorie: **Integration**
3. "Radiator Control MPC" suchen und **herunterladen**
4. **Home Assistant neustarten**
5. Einstellungen → Geräte & Dienste → **Integration hinzufügen** → "Radiator Control" suchen
6. Thermostat, Sensoren und Calibration-Entity auswählen → Fertig!

### Option B: Manuelle Installation

1. `custom_components/radiator_control/` in dein HA `config/custom_components/` kopieren
2. Home Assistant neustarten
3. Einstellungen → Geräte & Dienste → Integration hinzufügen → "Radiator Control"

### Option C: Standalone (CLI)

```bash
# 1. Setup
poetry install
cp config.yaml.example config.yaml  # Anpassen!

# 2. Verbindung testen
poetry run python main.py --test

# 3. Experiment-Übersicht lesen
poetry run python main.py --info

# 4. Systemidentifikation (~2.5h)
poetry run python main.py --experiment step

# 5. Regelung starten
poetry run python main.py

# 6. Nach ein paar Tagen: Performance prüfen
poetry run python main.py --stats
```

### Voraussetzungen

- Home Assistant 2024.1 oder neuer
- Ein Zigbee2MQTT-kompatibles Thermostat mit **Temperature Calibration** Entity (z.B. TuYa TS0601)
- Ein separater Temperatursensor (z.B. Aqara WSDCGQ11LM)
- Optional: Außentemperatur-Sensor, Fenster-Sensor (binary_sensor)

## 🎯 Was macht dieses Projekt?

Dieses System übernimmt die Regelung deiner Heizung, indem es das Thermostat "austrickst": Statt die Ventilstellung direkt zu steuern (was bei den meisten Smart-Thermostaten nicht möglich ist), manipulieren wir die Temperatur, die das Thermostat "sieht".

### Das Grundprinzip

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Echter Sensor ─────┐                                         │
│   (Raumtemperatur)   │                                         │
│                      ▼                                         │
│              ┌──────────────┐      ┌──────────────┐           │
│              │     MPC      │      │   Adaptives  │           │
│              │  Controller  │◄────►│    Modell    │           │
│              └──────────────┘      └──────────────┘           │
│                      │                                         │
│                      ▼                                         │
│              ┌──────────────┐                                  │
│              │  Berechne    │                                  │
│              │   Offset     │                                  │
│              └──────────────┘                                  │
│                      │                                         │
│                      ▼                                         │
│              ┌──────────────┐      ┌──────────────┐           │
│              │  Thermostat  │─────►│   Heizkörper │           │
│              │ (Black Box)  │      │              │           │
│              └──────────────┘      └──────────────┘           │
│                      ▲                                         │
│                      │                                         │
│              "Fake" Temperatur                                 │
│              (echte Temp + Offset)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Beispiel:**
- Echte Raumtemperatur: 19°C
- Solltemperatur: 21°C
- → System sagt dem Thermostat: "Es sind nur 17°C!" (Offset: -2°C)
- → Thermostat denkt es ist zu kalt → heizt mehr

## 🧠 Wie funktioniert die Regelung?

### 1. Model Predictive Control (MPC)

MPC ist ein fortschrittlicher Regelungsansatz, der **in die Zukunft schaut**:

```
Jetzt        +5min       +10min      +15min      ...    +4h
  │            │            │           │                │
  ▼            ▼            ▼           ▼                ▼
┌────┐      ┌────┐      ┌────┐      ┌────┐           ┌────┐
│19°C│ ───► │19.5│ ───► │20.1│ ───► │20.6│ ───► ... │21°C│
└────┘      └────┘      └────┘      └────┘           └────┘
   │
   └── "Welchen Offset brauche ich JETZT, damit ich in 4h bei 21°C bin?"
```

#### Warum 4 Stunden Horizont?

Heizkörper haben eine **erhebliche thermische Trägheit**:
- Der Heizkörper braucht Zeit zum Aufheizen (Wassermasse)
- Nach dem Ventilschluss strahlt er noch 20-30 Minuten Wärme ab (**Nachheizeffekt**)
- Die Raumtemperatur reagiert verzögert (Möbel, Wände speichern Wärme)

**Faustregel:** Prediction Horizon ≥ 2-3× Zeitkonstante (τ) des Raums

| System | Zeitkonstante τ | Empfohlener Horizon |
|--------|-----------------|---------------------|
| Schnell (Konvektoren, kleine Räume) | 60-90 min | 180 min (3h) |
| **Normal (Standard-Heizkörper)** | **90-150 min** | **240 min (4h)** |
| Träge (große Heizkörper, viel Masse) | 150-240 min | 360 min (6h) |
| Sehr träge (alte Gussheizkörper) | 200+ min | 480 min (8h) |

> ⚠️ **Bei zu kurzem Horizont:** MPC unterschätzt den Nachheizeffekt → Überschwingen!

**Der MPC-Algorithmus optimiert:**
```
min  Σ [ w_comfort * (T - T_soll)² + w_energie * offset² + w_glatt * Δoffset² ]
```

- **Komfort**: Temperatur möglichst nah am Sollwert
- **Energie**: Weniger heizen = weniger Kosten
- **Glätte**: Keine wilden Schwankungen (schont Ventil)

### 2. Adaptives Thermisches Modell

Das System **lernt** das Verhalten deines Raums:

```
dT/dt = (1/τ) * [ K_heiz * offset + K_außen * (T_außen - T) + K_fenster * fenster_offen * (T_außen - T) ]
```

**Parameter werden kontinuierlich angepasst (RLS - Recursive Least Squares):**

| Parameter | Bedeutung | Typischer Wert |
|-----------|-----------|----------------|
| τ (tau) | Zeitkonstante des Raums | 60-240 min |
| K_heiz | Heizungswirkung | 0.2-1.0 |
| K_außen | Einfluss Außentemperatur | 0.05-0.2 |
| K_fenster | Zusätzlicher Verlust bei offenem Fenster | 0.1-0.5 |

Das Modell passt sich an:
- Jahreszeiten
- Möbeländerungen
- Isolationsänderungen
- Unterschiedliche Heizkörperleistung

### 3. Fenster-Erkennung

Bei offenem Fenster hast du zwei Optionen (konfigurierbar):

**Option 1: Thermostat ausschalten** (`window_action: turn_off`) - **Empfohlen**
- ✅ Thermostat wird komplett auf "off" geschaltet
- ✅ Konfigurierbare Verzögerung (`window_off_delay_seconds`)
- ✅ Beim Schließen wird der vorherige Modus wiederhergestellt
- ✅ Kein unnötiges Heizen bei offenem Fenster

**Option 2: Maximaler Offset** (`window_action: offset`)
- ✅ Maximaler positiver Offset (Thermostat denkt es ist warm)
- ✅ Thermostat bleibt an, heizt aber nicht

Für beide Optionen gilt:
- ✅ Modell wird **nicht** mit diesen Daten trainiert (würde verfälschen)
- ✅ Daten werden trotzdem geloggt (für KI-Analysen)
- ✅ Automatische Wiederaufnahme wenn Fenster geschlossen

### 4. Solltemperatur aus Home Assistant

Die Solltemperatur wird **direkt vom Thermostat** in Home Assistant gelesen:
- Keine doppelte Konfiguration nötig
- Änderungen in der HA-App wirken sofort
- Unterstützt Zeitpläne und Automatisierungen in HA

## 📊 Datensammlung für KI-Training

Alle Daten werden für späteres Machine Learning gespeichert:

### Training Samples (RL-Ready)

```python
{
    "state": {
        "room_temp": 19.5,
        "outside_temp": 5.0,
        "target_temp": 21.0,
        "window_open": False,
        "previous_temp": 19.3,
        "previous_offset": -1.5,
        "heating_active": True
    },
    "action": {
        "offset": -2.0
    },
    "reward": 0.85,  # Komfort - Energie
    "next_state": {...}
}
```

### Export für Training

```bash
# Exportiere als Parquet für effizientes ML-Training
poetry run python -c "from src.database import Database; Database().export_training_data()"
```

## 🚀 Installation

### Voraussetzungen

- Python 3.10+
- Home Assistant mit:
  - Zigbee2MQTT (für Silvercrest Thermostat)
  - Long-Lived Access Token
  - Separater Temperatursensor (empfohlen)
  - Fensterkontakt (optional aber empfohlen)

### Setup

```bash
# Repository klonen
cd ~/radiator_control

# Poetry installieren (falls nicht vorhanden)
curl -sSL https://install.python-poetry.org | python3 -

# Dependencies installieren
poetry install

# Für ML/KI-Training zusätzlich:
poetry install --with ml
```

### Konfiguration

1. **Home Assistant Token erstellen:**
   - Settings → Long-Lived Access Tokens → Create Token

2. **config.yaml anpassen:**

```yaml
homeassistant:
  url: "http://homeassistant.local:8123"
  token: "YOUR_LONG_LIVED_ACCESS_TOKEN"

entities:
  thermostat: "climate.wohnzimmer_thermostat"
  temp_sensor: "sensor.wohnzimmer_temperatur"  # Genauer als Thermostat!
  window_sensor: "binary_sensor.fenster_wohnzimmer"
  outside_temp: "sensor.aussentemperatur"
```

3. **Verbindung testen:**

```bash
poetry run python main.py --test
```

## 🔧 Verwendung

### Befehlsübersicht

```bash
# Hilfe anzeigen
poetry run python main.py --help

# Verbindung testen
poetry run python main.py --test

# Info zu Experimenten anzeigen
poetry run python main.py --info

# Regelungs-Statistiken anzeigen
poetry run python main.py --stats
poetry run python main.py --stats --days 14  # Letzte 14 Tage

# Experiment-Ergebnisse anzeigen
poetry run python main.py --experiments

# Regelung starten
poetry run python main.py

# Docker
docker-compose up -d
```

### Systemidentifikation (empfohlen vor erstem Einsatz)

```bash
# Erst Übersicht lesen
poetry run python main.py --info

# Sprungantwort: ~2.5 Stunden, bestimmt τ und K_heiz
poetry run python main.py --experiment step

# PRBS: ~4 Stunden, robustere Identifikation
poetry run python main.py --experiment prbs

# Relay-Feedback: ~2-3 Stunden, für PID-Tuning
poetry run python main.py --experiment relay
```

**Die Experimente sind interaktiv:**
- Klare Erklärung was passiert
- Geschätzte Dauer
- Fortschrittsanzeige
- Warnungen (Fenster geschlossen halten!)
- Ergebnisse werden automatisch ins Modell übernommen

### Experiment-Ergebnisse anzeigen

```bash
poetry run python main.py --experiments
```

Zeigt alle durchgeführten Experimente mit:
- Identifizierte Parameter (τ, k_heater, etc.)
- Zeitpunkt und Dauer
- Status (erfolgreich/abgebrochen)

**Parameter werden automatisch ins Modell übernommen!**

### Training-Daten exportieren

```bash
poetry run python -c "
from src.database import Database
db = Database()
print(db.get_training_stats())
db.export_training_data('data/training_data.parquet')
"
```

## 📈 Optimierungsprozess

### Phase 1: Systemidentifikation (Tag 1)

```bash
poetry run python main.py --experiment step
```

Führt einen 3-stündigen Test durch:
1. Offset auf 0°C stabilisieren
2. Sprung auf -3°C (mehr heizen)
3. Temperaturanstieg beobachten
4. τ und K_heiz identifizieren

### Phase 2: Online-Lernen (kontinuierlich)

Das adaptive Modell verbessert sich mit jeder Minute:

```
Woche 1: RMSE = 0.3°C (Modell noch ungenau)
Woche 2: RMSE = 0.15°C (lernt das Raumverhalten)
Woche 4: RMSE = 0.08°C (sehr genaue Vorhersagen)
```

**Fortschritt überwachen:**

```bash
# Tägliche Statistiken mit RMSE, MAE, Komfort-Quote
poetry run python main.py --stats
```

Beispiel-Output:
```
══════════════════════════════════════════════════════════════════════
                     📊 REGELUNGS-STATISTIK                          
══════════════════════════════════════════════════════════════════════

Performance-Übersicht (letzte 7 Tage):
──────────────────────────────────────────────────────────────────────

  RMSE (Wurzel mittlerer quadr. Fehler):  0.234°C  ✅ Gut
  MAE (Mittlerer absoluter Fehler):       0.189°C
  Komfort-Quote (±0.5°C vom Sollwert):    87.3%
  
Trend:
  📈 VERBESSERT (+15.2% gegenüber Vorwoche)

Tägliche Übersicht:
Datum           RMSE      MAE   Komfort  Samples    Offset
2026-01-30    0.198°   0.156°    91.2%     1440    -1.23°
2026-01-29    0.245°   0.201°    85.4%     1440    -1.45°
...
```

**Bewertung des RMSE:**
| RMSE | Bewertung |
|------|----------|
| < 0.3°C | 🌟 Exzellent |
| 0.3-0.5°C | ✅ Gut |
| 0.5-0.8°C | ⚠️ Verbesserungswürdig |
| > 0.8°C | ❌ Schlecht |

### Phase 3: MPC-Tuning (optional)

```yaml
# config.yaml - Gewichte anpassen
mpc:
  weight_comfort: 1.0    # Höher = weniger Temperaturschwankungen
  weight_energy: 0.1     # Höher = mehr Energiesparen
  weight_smoothness: 0.05  # Höher = sanftere Übergänge
```

### Phase 4: KI/RL-Training (fortgeschritten)

Mit gesammelten Daten kannst du später trainieren:

```python
import pandas as pd
from stable_baselines3 import SAC

# Lade Daten
df = pd.read_parquet("data/training_data.parquet")

# Erstelle Gym Environment (siehe src/rl_controller.py)
# Trainiere RL-Agent
# ...
```

## 🗂️ Projektstruktur

```
radiator_control/
├── config.yaml           # Konfiguration (ohne Solltemperatur - kommt aus HA!)
├── pyproject.toml        # Poetry Dependencies
├── main.py               # Hauptprogramm & CLI
├── repository.yaml       # HA Add-on Repository Manifest
├── src/
│   ├── ha_client.py      # Home Assistant API
│   ├── database.py       # SQLite + Training-Logs + Statistiken
│   ├── model.py          # Adaptives thermisches Modell (RLS)
│   ├── mpc_controller.py # MPC Optimierung (CVXPY/OSQP)
│   ├── experiments.py    # Interaktive Systemidentifikation
│   └── utils.py          # Hilfsfunktionen
├── radiator-control/        # Home Assistant Add-on
│   ├── config.yaml       # Add-on Manifest
│   ├── Dockerfile        # Container für HA
│   ├── DOCS.md           # Add-on Dokumentation
│   ├── rootfs/           # Container Root-Filesystem
│   │   └── run.sh        # Startskript
│   ├── templates/        # Web-UI Templates
│   └── static/           # CSS & JavaScript
├── data/
│   ├── measurements.db   # SQLite: Messungen, Experimente, Training-Samples
│   ├── model.json        # Gespeichertes Modell (wird automatisch geladen)
│   └── training_data.parquet  # Export für ML
├── Dockerfile            # Multi-stage Docker Build (Standalone)
├── docker-compose.yml    # Container-Deployment
└── logs/
    └── radiator_control.log
```

## 🏠 Home Assistant Add-on

Das Add-on bietet eine **Web-UI** direkt in Home Assistant:

### Features

- **Dashboard**: Aktuelle Temperaturen, Offset, HVAC-Modus, Fenster-Status
- **Quick Actions**: Thermostat ein/ausschalten, Regelung starten/stoppen
- **Temperaturverlauf**: Live-Graph mit Raum-, Soll- und Außentemperatur
- **Experimente**: Mit einem Klick starten
- **Statistiken**: RMSE, MAE, Komfort-Quote, Trend, tägliche Auswertung
- **Modell-Info**: Aktuelle Parameter (τ, K_heiz, K_außen), Modell-RMSE
- **Einstellungen**: MPC-Gewichte, Offset-Limits, Fenster-Aktion direkt in der UI
- **Climate Entity**: Optional für better-thermostat-ui-card Kompatibilität

### HA-Sensoren

Werden automatisch erstellt:
- `sensor.radiator_control_rmse` - Modellgenauigkeit
- `sensor.radiator_control_offset` - Aktueller Offset
- `sensor.radiator_control_mode` - Aktueller Modus
- `climate.radiator_mpc_control` - Climate Entity (optional)

### Konfiguration

| Option | Beschreibung | Beispiel |
|--------|--------------|----------|
| `thermostat_entity` | Climate-Entity | `climate.wohnzimmer` |
| `temp_sensor_entity` | Genauer Sensor | `sensor.wohnzimmer_temp` |
| `window_sensor_entity` | Fensterkontakt | `binary_sensor.fenster_wz` |
| `outside_temp_entity` | Außentemperatur | `sensor.aussentemperatur` |
| `window_action` | Bei offenem Fenster | `turn_off` oder `offset` |
| `window_off_delay_seconds` | Verzögerung | `30` |
| `create_climate_entity` | Climate für better-thermostat-ui-card | `true` |

### Web-UI

```
┌─────────────────────────────────────────────────────────────────────┐
│  🌡️ Radiator Control                           [Regelung aktiv]    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🏠 20.5°C   🎯 21.0°C   📊 -1.2°C   🌤️ 5°C   🔥 Heizen   ✅ Zu    │
│  Raum       Soll        Offset      Außen    HVAC      Fenster     │
│                                                                     │
│  [▶️ Regelung starten]  [⏹️ Stoppen]  [🔥 Heizung AN]  [❄️ AUS]   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  📈 Temperaturverlauf (24h)                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │     ───── Raum   ───── Soll   ───── Außen                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  ℹ️ Status        | 🧠 Modell       | 📊 Performance | 🔌 Entities │
│  Mode: Regelung  | τ: 120 min     | RMSE: 0.23°C   | Thermostat  │
│  HVAC: Heizen    | K_heiz: 0.45   | Komfort: 87%   | Temp Sensor │
│  Heizung: Ja     | RMSE: 0.0012   | Trend: 📈      | Window      │
│  Fenster: Zu     | Updates: 1.2k  | Samples: 5.4k  | Outside     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  ⚙️ Einstellungen                                                   │
│  MPC: Horizont 120min, Gewichte anpassbar                          │
│  Offset: -5°C bis +5°C                                             │
│  Fenster: Thermostat ausschalten nach 30s                          │
│  Modell: Vergessensrate 0.98                                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  🏠 Climate Entity für better-thermostat-ui-card                    │
│  Entity ID: climate.radiator_mpc_control  [📋 Kopieren]            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🐳 Docker (Standalone)

```bash
# Bauen und starten
docker-compose up -d

# Logs ansehen
docker-compose logs -f

# Stoppen
docker-compose down
```

## ❓ FAQ

### Warum nicht direkt das Ventil steuern?

Die meisten Smart-Thermostate (inkl. Silvercrest) erlauben keine direkte Ventilsteuerung. Sie haben einen internen Regler, der auf die gemessene Temperatur reagiert. Wir "überlisten" diesen Regler.

### Wie genau ist die Regelung?

Nach einigen Tagen Lernzeit: ±0.3°C um den Sollwert.

Überprüfe mit `python main.py --stats` - dort siehst du RMSE und Komfort-Quote.

### Funktioniert das mit mehreren Räumen?

Aktuell für einen Raum ausgelegt. Erweiterung auf mehrere Räume möglich durch separate Instanzen oder Refactoring.

### Was passiert bei Stromausfall?

Das Thermostat fällt auf seinen internen Regler zurück (funktioniert weiter, nur nicht optimiert). Beim Neustart lädt das System das gespeicherte Modell.

### Wo sehe ich die Experiment-Ergebnisse?

```bash
poetry run python main.py --experiments
```

Alle Ergebnisse werden persistent in der SQLite-Datenbank gespeichert. Die identifizierten Parameter werden automatisch ins Modell übernommen.

### Wie sehe ich ob die Regelung besser wird?

```bash
poetry run python main.py --stats --days 14
```

Zeigt RMSE-Trend: `📈 VERBESSERT` oder `📉 VERSCHLECHTERT` im Vergleich zur Vorwoche.

### Wie viel Energie spart das?

Abhängig von der Situation. Typisch 10-20% durch:
- Prädiktives Heizen (nicht zu früh, nicht zu spät)
- Optimale Balance Komfort/Energie
- Schnelles Abschalten bei offenem Fenster

## 📜 Lizenz

MIT License
