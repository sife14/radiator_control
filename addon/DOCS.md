# Radiator Control - Home Assistant Add-on Repository

Dieses Repository enthält das Radiator Control MPC Add-on für Home Assistant.

## Installation

### Methode 1: Als lokales Add-on (Entwicklung)

1. Kopiere den `addon/` Ordner nach `/addons/radiator_control/` auf deinem Home Assistant System:
   ```bash
   scp -r addon/ root@homeassistant.local:/addons/radiator_control/
   ```

2. In Home Assistant:
   - Einstellungen → Add-ons → Add-on Store
   - Oben rechts: ⋮ → Repositories neu laden
   - Das Add-on sollte unter "Lokale Add-ons" erscheinen

### Methode 2: Als Add-on Repository

1. Erstelle ein GitHub Repository mit dieser Struktur:
   ```
   repository.yaml
   radiator_control/
   ├── config.yaml
   ├── Dockerfile
   ├── ...
   ```

2. In Home Assistant:
   - Einstellungen → Add-ons → Add-on Store
   - ⋮ → Repositories → Repository hinzufügen
   - URL: `https://github.com/DEIN_USERNAME/ha-addons`

## Konfiguration

Nach der Installation in der Add-on Konfiguration:

### Entities

| Option | Beschreibung | Beispiel |
|--------|--------------|----------|
| `thermostat_entity` | Climate-Entity des Thermostats | `climate.wohnzimmer` |
| `temp_sensor_entity` | Genauer Temperatursensor | `sensor.wohnzimmer_temp` |
| `window_sensor_entity` | Fensterkontakt (optional) | `binary_sensor.fenster_wz` |
| `outside_temp_entity` | Außentemperatursensor | `sensor.aussentemperatur` |
| `temp_calibration_entity` | Kalibrierungs-Offset Entity (optional) | `number.thermostat_calibration` |

### Fenster-Verhalten

| Option | Beschreibung | Standardwert |
|--------|--------------|--------------|
| `window_action` | Aktion bei offenem Fenster: `turn_off` (Thermostat aus) oder `offset` (nur Offset) | `turn_off` |
| `window_off_delay_seconds` | Verzögerung in Sekunden bevor Thermostat aus | `30` |

### MPC Parameter

| Option | Beschreibung | Standardwert |
|--------|--------------|--------------|
| `mpc_horizon_minutes` | Vorhersagehorizont (siehe unten!) | `240` |
| `mpc_control_horizon` | Steuerhorizont | `15` |
| `mpc_weight_comfort` | Gewichtung Komfort | `1.0` |
| `mpc_weight_energy` | Gewichtung Energie | `0.1` |
| `mpc_weight_smoothness` | Gewichtung Glätte | `0.05` |

#### Prediction Horizon richtig wählen

Der Heizkörper strahlt nach dem Ausschalten noch lange Wärme ab (**Nachheizeffekt**). Der Horizon muss lang genug sein, um diese Trägheit zu erfassen!

**Faustregel:** Horizon ≥ 2-3× Zeitkonstante (τ) des Raums

| Dein System | Empfohlener Horizon |
|-------------|---------------------|
| Konvektoren, kleine Räume | 180 min |
| **Normale Heizkörper** | **240 min (Standard)** |
| Große Heizkörper, viel Masse | 360 min |
| Alte Gussheizkörper | 480 min |

> ⚠️ Wenn die Temperatur nach Erreichen des Sollwerts noch weiter steigt (Überschwingen), erhöhe den Horizon!

### Control Parameter

| Option | Beschreibung | Standardwert |
|--------|--------------|--------------|
| `offset_min` | Minimaler Offset | `-5.0` |
| `offset_max` | Maximaler Offset | `5.0` |
| `sample_time_seconds` | Abtastzeit | `60` |

### Modell Parameter

| Option | Beschreibung | Standardwert |
|--------|--------------|--------------|
| `model_initial_tau` | Initiale Zeitkonstante (min) | `120` |
| `model_initial_k_heater` | Initialer Heizungsfaktor | `0.5` |
| `model_forgetting_factor` | RLS Vergessensrate | `0.98` |

### Climate Entity (better-thermostat-ui-card)

| Option | Beschreibung | Standardwert |
|--------|--------------|--------------|
| `create_climate_entity` | Climate Entity erstellen | `true` |
| `climate_entity_name` | Anzeigename | `Radiator Control` |

## Features

### Web-UI

Die Web-UI ist direkt in Home Assistant über die Seitenleiste erreichbar:

- **Status-Karten**: Raumtemperatur, Soll, Offset, Außen, HVAC-Modus, Fenster
- **Quick Actions**: Regelung starten/stoppen, Thermostat ein/aus
- **Temperatur-Graph**: Live-Verlauf der letzten 24 Stunden
- **Info-Boxen**: Status, Modell-Parameter, Performance, Entities
- **Experimente**: Systemidentifikation starten
- **Statistiken**: Tägliche RMSE, MAE, Komfort-Quote
- **Einstellungen**: MPC, Offset, Fenster, Modell-Parameter anpassen

### HA-Sensoren

Werden automatisch erstellt:
- `sensor.radiator_control_rmse` - Modellgenauigkeit
- `sensor.radiator_control_offset` - Aktueller Offset
- `sensor.radiator_control_mode` - Aktueller Modus

### Climate Entity (optional)

Für better-thermostat-ui-card:
- `climate.radiator_mpc_control` - Virtuelle Climate-Entity
- Zeigt aktuelle/Soll-Temperatur, HVAC-Modus, Offset, Fenster-Status

### API Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/status` | GET | Aktueller Status |
| `/api/stats` | GET | Statistiken |
| `/api/model` | GET | Modell-Parameter |
| `/api/experiments` | GET | Experiment-Liste |
| `/api/config` | GET/POST | Konfiguration lesen/schreiben |
| `/api/control/start` | POST | Regelung starten |
| `/api/control/stop` | POST | Regelung stoppen |
| `/api/thermostat/on` | POST | Thermostat einschalten |
| `/api/thermostat/off` | POST | Thermostat ausschalten |
| `/api/model/reset` | POST | Modell zurücksetzen |
| `/api/experiment/start` | POST | Experiment starten |
| `/api/experiment/stop` | POST | Experiment abbrechen |

## Entwicklung

```bash
# Lokal testen mit Docker
cd addon
docker build -t radiator-control --build-arg BUILD_FROM=python:3.11-alpine .
docker run -p 5000:5000 radiator-control
```
