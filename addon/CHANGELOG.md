# Changelog

## [1.1.0] - 2025-01-30

### Added
- **Thermostat Ein/Aus**: Bei offenem Fenster wird das Thermostat jetzt komplett ausgeschaltet (statt nur Offset)
- **Konfigurierbare Fenster-Aktion**: `turn_off` (Thermostat aus) oder `offset` (nur Offset ändern)
- **Verzögerung bei Fenster**: Konfigurierbare Wartezeit bevor Thermostat ausgeschaltet wird
- **Erweiterte WebUI**:
  - HVAC-Modus Anzeige
  - Fenster-Status Karte
  - Quick Actions: Thermostat ein/aus
  - Temperaturverlauf-Graph (24h)
  - Erweiterte Info-Boxen (Status, Modell, Performance, Entities)
  - Einstellungs-Panel (MPC, Offset, Fenster, Modell)
  - Modell-Reset Funktion
- **Climate Entity**: Optionale Climate-Entity für better-thermostat-ui-card Kompatibilität
- **Mehr Konfigurationsoptionen**: Modell-Parameter, MPC Control Horizon
- **API Erweiterungen**: `/api/thermostat/on`, `/api/thermostat/off`, `/api/config`, `/api/model/reset`

### Changed
- Verbesserte Fenster-Logik mit gespeichertem HVAC-Mode
- Status enthält jetzt `hvac_mode` und `heating_active`
- Entities werden in der UI angezeigt

### Fixed
- Syntax-Fehler in ha_client.py (fehlende Newlines zwischen Methoden)

## [1.0.0] - 2025-01-30

### Added
- Initial release
- MPC-basierte Heizungsregelung
- Adaptives thermisches Modell mit RLS
- Web-UI für Experimente und Statistiken
- Systemidentifikation (Sprungantwort, PRBS, Relay-Feedback)
- HA-Sensoren für RMSE, Offset, Mode
- Fenster-Erkennung mit automatischer Heizungsabschaltung
- Tägliche Statistiken mit RMSE und Komfort-Quote
