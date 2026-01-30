"""
Experiment Runner
=================
Automatische Systemidentifikation durch Sprungantwort und PRBS-Tests.
"""

import asyncio
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import json

from .ha_client import HomeAssistantClient, SensorData
from .database import Database, Measurement
from .model import ThermalModel, ThermalModelParams, identify_from_step_response

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    STEP_RESPONSE = "step_response"
    PRBS = "prbs"
    RELAY = "relay"


@dataclass
class ExperimentConfig:
    """Konfiguration für Experimente."""
    
    # Sprungantwort
    step_offset: float = -3.0          # Offset-Sprung [°C] (negativ = mehr heizen)
    step_duration_minutes: int = 120   # Beobachtungsdauer [min] - 2 Stunden reichen meist
    
    # PRBS (Pseudo-Random Binary Sequence)
    prbs_amplitude: float = 2.0        # ±amplitude [°C]
    prbs_min_duration: int = 15        # Min. Haltedauer pro Zustand [min]
    prbs_total_duration: int = 240     # Gesamtdauer [min] - 4 Stunden
    
    # Relay-Feedback (für Autotuning)
    relay_hysteresis: float = 0.3      # Hysterese [°C]
    relay_max_duration: int = 180      # Max. Dauer [min] - 3 Stunden
    
    # Allgemein
    sample_interval: int = 60          # Messintervall [s]
    pre_experiment_settle: int = 15    # Wartezeit vor Start [min]


@dataclass
class ExperimentResult:
    """Ergebnis eines Experiments."""
    
    experiment_type: ExperimentType
    start_time: datetime
    end_time: datetime
    
    # Rohdaten
    times: np.ndarray          # Zeit relativ zum Start [min]
    temps: np.ndarray          # Raumtemperaturen
    offsets: np.ndarray        # Angewendete Offsets
    outside_temps: np.ndarray  # Außentemperaturen
    
    # Identifizierte Parameter (falls verfügbar)
    identified_params: Optional[ThermalModelParams] = None
    
    # Metriken
    metrics: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            'experiment_type': self.experiment_type.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'times': self.times.tolist(),
            'temps': self.temps.tolist(),
            'offsets': self.offsets.tolist(),
            'outside_temps': self.outside_temps.tolist(),
            'identified_params': self.identified_params.to_dict() if self.identified_params else None,
            'metrics': self.metrics,
        }


class ExperimentRunner:
    """
    Führt Experimente zur Systemidentifikation durch.
    
    Unterstützte Experimente:
    1. Sprungantwort: Sprung im Offset, beobachte Reaktion
    2. PRBS: Pseudo-Random Binary Sequence für Frequenzanalyse
    3. Relay-Feedback: Für Ziegler-Nichols Autotuning
    """
    
    def __init__(
        self,
        ha_client: HomeAssistantClient,
        database: Database,
        config: Optional[ExperimentConfig] = None,
    ):
        self.ha_client = ha_client
        self.db = database
        self.config = config or ExperimentConfig()
        
        self._running = False
        self._current_experiment: Optional[ExperimentType] = None
        self._stop_event = asyncio.Event()
        
        # Callbacks
        self._progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None
    
    def set_progress_callback(self, callback: Callable[[float, str], Awaitable[None]]):
        """Setzt Callback für Fortschrittsmeldungen."""
        self._progress_callback = callback
    
    async def _report_progress(self, progress: float, message: str):
        """Meldet Fortschritt."""
        logger.info(f"Experiment progress: {progress*100:.1f}% - {message}")
        if self._progress_callback:
            await self._progress_callback(progress, message)
    
    def _print_banner(self, text: str, char: str = "="):
        """Druckt einen auffälligen Banner."""
        width = 70
        print(f"\n{char * width}")
        print(f"{text.center(width)}")
        print(f"{char * width}\n")
    
    def _print_status(self, message: str, icon: str = "ℹ️"):
        """Druckt eine Statusmeldung."""
        print(f"{icon}  {message}")
    
    def _print_instruction(self, message: str):
        """Druckt eine Anweisung für den Nutzer."""
        print(f"👉 {message}")
    
    def _print_warning(self, message: str):
        """Druckt eine Warnung."""
        print(f"⚠️  {message}")
    
    def _print_success(self, message: str):
        """Druckt eine Erfolgsmeldung."""
        print(f"✅ {message}")
    
    def _print_progress_bar(self, progress: float, width: int = 40):
        """Druckt einen Fortschrittsbalken."""
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        print(f"\r[{bar}] {progress*100:5.1f}%", end="", flush=True)
    
    def show_experiment_overview(self):
        """Zeigt eine Übersicht aller verfügbaren Experimente."""
        self._print_banner("🔬 SYSTEMIDENTIFIKATION - ÜBERSICHT", "═")
        
        print("""
Was ist Systemidentifikation?
─────────────────────────────
Die Systemidentifikation ermittelt, wie dein Raum auf Heizung reagiert.
Jeder Raum ist anders: Größe, Isolierung, Fenster, Außenwände...
Mit den gemessenen Daten kann der Controller optimal regeln.

""")
        
        cfg = self.config
        step_duration = cfg.pre_experiment_settle + cfg.step_duration_minutes
        
        print("Verfügbare Experimente:")
        print("═" * 70)
        
        print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│  1. SPRUNGANTWORT (step_response)                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ⏱️  Dauer: ca. {step_duration} Minuten ({step_duration/60:.1f} Stunden)                              │
│                                                                      │
│  📋 Was passiert:                                                    │
│     • Der Heizungs-Offset wird um {abs(cfg.step_offset)}°C reduziert (mehr heizen)        │
│     • Die Raumtemperatur steigt langsam an                          │
│     • Aus der Anstiegskurve werden Modellparameter berechnet        │
│                                                                      │
│  🎯 Wofür geeignet:                                                  │
│     • Erster Test eines neuen Raums                                 │
│     • Grundlegende Parameterbestimmung                              │
│     • Einfach und zuverlässig                                       │
│                                                                      │
│  💡 Empfehlung: Starte hiermit!                                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  2. PRBS-TEST (prbs)                                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ⏱️  Dauer: ca. {cfg.prbs_total_duration} Minuten ({cfg.prbs_total_duration/60:.1f} Stunden)                              │
│                                                                      │
│  📋 Was passiert:                                                    │
│     • Zufällige Wechsel zwischen +{cfg.prbs_amplitude}°C und -{cfg.prbs_amplitude}°C Offset          │
│     • Wechsel alle {cfg.prbs_min_duration} Minuten (mindestens)                         │
│     • Reichhaltige Daten für Modellidentifikation                   │
│                                                                      │
│  🎯 Wofür geeignet:                                                  │
│     • Präzisere Modellparameter                                     │
│     • Daten für Machine Learning / KI-Training                      │
│     • Dynamisches Verhalten verstehen                               │
│                                                                      │
│  💡 Tipp: Nach Sprungantwort für bessere Daten                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  3. RELAY-FEEDBACK (relay)                                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ⏱️  Dauer: ca. {cfg.relay_max_duration} Minuten ({cfg.relay_max_duration/60:.1f} Stunden) - kann früher enden             │
│                                                                      │
│  📋 Was passiert:                                                    │
│     • Temperatur schwingt um Sollwert                               │
│     • Automatischer Wechsel bei Über-/Unterschreiten                │
│     • Aus Schwingung werden PID-Parameter berechnet                 │
│                                                                      │
│  🎯 Wofür geeignet:                                                  │
│     • PID-Regler Tuning (Ziegler-Nichols)                           │
│     • Alternative zur Sprungantwort                                 │
│     • Selbstschwingendes System analysieren                         │
│                                                                      │
│  💡 Fortgeschritten - nicht unbedingt nötig für MPC                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
""")
        
        print("Wichtige Hinweise:")
        print("─" * 70)
        print("""
  ⚠️  Während der Experimente:
     • Fenster und Türen GESCHLOSSEN halten
     • Keine zusätzlichen Wärmequellen einschalten
     • Raum möglichst nicht betreten (Körperwärme!)
     • Außentemperatur sollte relativ konstant sein
     
  🌡️  Beste Bedingungen:
     • Nachts (wenig Störungen)
     • Bewölkter Tag (keine Sonneneinstrahlung)
     • Außentemperatur mindestens 5°C unter Raumtemperatur
     
  ❌ Abbrechen:
     • Jederzeit mit Strg+C möglich
     • Heizung wird automatisch auf Normalbetrieb zurückgesetzt
""")
        
        print("═" * 70)
        print()
    
    async def _collect_sample(self, offset: float) -> SensorData:
        """Sammelt einen Messpunkt."""
        # Offset setzen
        await self.ha_client.set_temperature_offset(offset)
        
        # Kurz warten
        await asyncio.sleep(2)
        
        # Sensoren lesen
        data = await self.ha_client.read_all_sensors()
        
        return data
    
    async def _save_measurement(
        self,
        data: SensorData,
        offset: float,
        target_temp: float,
        mode: str,
    ):
        """Speichert Messung in Datenbank."""
        measurement = Measurement(
            timestamp=data.timestamp,
            room_temp=data.room_temp,
            outside_temp=data.outside_temp,
            window_open=data.window_open,
            heating_active=data.heating_active,
            control_offset=offset,
            target_temp=target_temp,
            mode=mode,
        )
        self.db.insert_measurement(measurement)
    
    def stop(self):
        """Stoppt laufendes Experiment."""
        self._stop_event.set()
        logger.info("Experiment stop requested")
    
    # -------------------------------------------------------------------------
    # Sprungantwort
    # -------------------------------------------------------------------------
    
    async def run_step_response(
        self,
        initial_offset: float = 0.0,
        step_offset: Optional[float] = None,
        duration_minutes: Optional[int] = None,
        interactive: bool = True,
    ) -> ExperimentResult:
        """
        Führt Sprungantwort-Experiment durch.
        
        WAS PASSIERT:
        1. System stabilisiert sich bei normalem Betrieb (15 min)
        2. Heizung wird auf Maximum gestellt (Offset = -3°C)
        3. Temperaturanstieg wird gemessen (2 Stunden)
        4. Aus dem Anstieg werden τ (Zeitkonstante) und K (Verstärkung) berechnet
        
        DAUER: ca. 2,5 Stunden
        
        Args:
            initial_offset: Start-Offset [°C]
            step_offset: End-Offset [°C] (default: -3°C für maximales Heizen)
            duration_minutes: Beobachtungsdauer [min]
            interactive: Wenn True, zeigt Fortschritt und Anweisungen
            
        Returns:
            ExperimentResult mit Messdaten und identifizierten Parametern
        """
        cfg = self.config
        step_offset = step_offset if step_offset is not None else cfg.step_offset
        duration = duration_minutes or cfg.step_duration_minutes
        
        total_duration = cfg.pre_experiment_settle + duration
        
        if interactive:
            self._print_banner("🔬 SYSTEMIDENTIFIKATION: SPRUNGANTWORT")
            
            print("Was wird gemacht:")
            print("─" * 50)
            print("""
Dieses Experiment bestimmt, wie dein Raum auf Heizung reagiert.
Wir messen zwei wichtige Parameter:

  τ (Tau) = Zeitkonstante des Raums
            Wie schnell reagiert der Raum auf Heizung?
            Typisch: 60-240 Minuten
            
  K       = Verstärkung
            Wie stark wirkt sich die Heizung aus?
            
Diese Parameter braucht der MPC-Regler für genaue Vorhersagen.
""")
            
            print("Ablauf:")
            print("─" * 50)
            print(f"""
  Phase 1: Stabilisierung     ({cfg.pre_experiment_settle} Minuten)
           → Heizung auf Normalbetrieb
           → Temperatur stabilisiert sich
           
  Phase 2: Heizphase          ({duration} Minuten)  
           → Heizung auf Maximum
           → Temperaturanstieg wird gemessen
           
  Phase 3: Auswertung         (wenige Sekunden)
           → Parameter werden berechnet
           → Modell wird gespeichert
""")
            
            print("Wichtige Hinweise:")
            print("─" * 50)
            self._print_warning("FENSTER MÜSSEN GESCHLOSSEN BLEIBEN!")
            self._print_warning("Raum sollte nicht betreten werden (verfälscht Messung)")
            self._print_warning("Keine anderen Wärmequellen (Ofen, Kerzen, etc.)")
            print()
            
            estimated_end = datetime.now() + timedelta(minutes=total_duration)
            self._print_status(f"Geschätzte Dauer: {total_duration} Minuten ({total_duration/60:.1f} Stunden)")
            self._print_status(f"Voraussichtliches Ende: {estimated_end.strftime('%H:%M Uhr')}")
            print()
            
            self._print_instruction("Drücke ENTER um zu starten (oder Ctrl+C zum Abbrechen)")
            try:
                input()
            except KeyboardInterrupt:
                print("\n❌ Abgebrochen.")
                return None
        
        self._running = True
        self._current_experiment = ExperimentType.STEP_RESPONSE
        self._stop_event.clear()
        
        # Experiment in DB registrieren
        exp_id = self.db.start_experiment(
            name=f"Step Response {datetime.now().strftime('%Y%m%d_%H%M')}",
            exp_type="step_response",
            parameters={
                'initial_offset': initial_offset,
                'step_offset': step_offset,
                'duration_minutes': duration,
            }
        )
        
        start_time = datetime.now()
        times = []
        temps = []
        offsets = []
        outside_temps = []
        
        try:
            if interactive:
                self._print_banner("Phase 1: Stabilisierung", "─")
                self._print_status(f"Setze Offset auf {initial_offset}°C (Normalbetrieb)")
                self._print_status(f"Warte {cfg.pre_experiment_settle} Minuten auf stabile Temperatur...")
                print()
            
            await self._report_progress(0, "Starting step response experiment")
            
            # Phase 1: Stabilisierung bei initial_offset
            settle_samples = cfg.pre_experiment_settle * 60 // cfg.sample_interval
            initial_temp = None
            
            for i in range(settle_samples):
                if self._stop_event.is_set():
                    raise asyncio.CancelledError("Experiment stopped")
                
                data = await self._collect_sample(initial_offset)
                await self._save_measurement(data, initial_offset, 21.0, 'experiment')
                
                if initial_temp is None:
                    initial_temp = data.room_temp
                
                if interactive:
                    progress = i / settle_samples
                    elapsed = (i * cfg.sample_interval) // 60
                    remaining = cfg.pre_experiment_settle - elapsed
                    self._print_progress_bar(progress)
                    print(f"  T={data.room_temp:.1f}°C | noch {remaining} min", end="")
                
                await asyncio.sleep(cfg.sample_interval)
            
            if interactive:
                print()  # Neue Zeile nach Fortschrittsbalken
                self._print_success(f"Stabilisierung abgeschlossen. Temperatur: {data.room_temp:.1f}°C")
            
            # Phase 2: Sprung und Beobachtung
            if interactive:
                self._print_banner("Phase 2: Heizphase", "─")
                self._print_status(f"Setze Offset auf {step_offset}°C (MAXIMALES HEIZEN)")
                self._print_status(f"Beobachte Temperaturanstieg für {duration} Minuten...")
                self._print_warning("Jetzt bitte den Raum nicht mehr betreten!")
                print()
            
            await self._report_progress(0.1, f"Step applied: {initial_offset}°C → {step_offset}°C")
            
            step_start = datetime.now()
            step_start_temp = data.room_temp
            total_samples = duration * 60 // cfg.sample_interval
            
            for i in range(total_samples):
                if self._stop_event.is_set():
                    raise asyncio.CancelledError("Experiment stopped")
                
                data = await self._collect_sample(step_offset)
                await self._save_measurement(data, step_offset, 21.0, 'experiment')
                
                # Fenster-Check
                if data.window_open:
                    if interactive:
                        print()
                        self._print_warning("FENSTER WURDE GEÖFFNET! Experiment wird abgebrochen.")
                        self._print_instruction("Bitte Fenster schließen und Experiment neu starten.")
                    raise asyncio.CancelledError("Window opened during experiment")
                
                # Daten sammeln
                elapsed = (datetime.now() - step_start).total_seconds() / 60
                times.append(elapsed)
                temps.append(data.room_temp)
                offsets.append(step_offset)
                outside_temps.append(data.outside_temp or 5.0)
                
                if interactive:
                    progress = i / total_samples
                    temp_change = data.room_temp - step_start_temp
                    elapsed_min = int(elapsed)
                    remaining_min = duration - elapsed_min
                    
                    self._print_progress_bar(progress)
                    print(f"  T={data.room_temp:.1f}°C (Δ{temp_change:+.1f}°C) | noch {remaining_min} min", end="")
                
                await asyncio.sleep(cfg.sample_interval)
            
            if interactive:
                print()  # Neue Zeile
                final_temp_change = temps[-1] - step_start_temp
                self._print_success(f"Heizphase abgeschlossen. Temperaturanstieg: {final_temp_change:+.1f}°C")
            
            # Phase 3: Auswertung
            if interactive:
                self._print_banner("Phase 3: Auswertung", "─")
                self._print_status("Analysiere Sprungantwort...")
            
            await self._report_progress(0.9, "Analyzing step response")
            
            times = np.array(times)
            temps = np.array(temps)
            offsets = np.array(offsets)
            outside_temps = np.array(outside_temps)
            
            # Parameter identifizieren
            identified_params = identify_from_step_response(
                times=times,
                temps=temps,
                offset_step=step_offset - initial_offset,
                outside_temp=np.mean(outside_temps),
            )
            
            # Metriken berechnen
            time_constant = self._find_time_constant(times, temps)
            settling_time = self._find_settling_time(times, temps, 0.95)
            
            metrics = {
                'temp_start': temps[0],
                'temp_end': temps[-1],
                'temp_change': temps[-1] - temps[0],
                'time_constant_63': time_constant,
                'settling_time_95': settling_time,
            }
            
            await self._report_progress(1.0, "Experiment completed")
            
            result = ExperimentResult(
                experiment_type=ExperimentType.STEP_RESPONSE,
                start_time=start_time,
                end_time=datetime.now(),
                times=times,
                temps=temps,
                offsets=offsets,
                outside_temps=outside_temps,
                identified_params=identified_params,
                metrics=metrics,
            )
            
            # In DB speichern
            self.db.end_experiment(exp_id, result.to_dict())
            
            # Ergebnis anzeigen
            if interactive:
                self._print_banner("🎉 ERGEBNIS", "═")
                
                print("Identifizierte Parameter:")
                print("─" * 50)
                p = identified_params
                print(f"""
  τ (Zeitkonstante)     = {p.tau:.0f} Minuten
                          → Dein Raum reagiert {"schnell" if p.tau < 90 else "normal" if p.tau < 180 else "langsam"} auf Heizung
                          
  K_heiz (Verstärkung)  = {p.k_heater:.2f}
                          → {"Starke" if p.k_heater > 0.6 else "Normale" if p.k_heater > 0.3 else "Schwache"} Heizwirkung
                          
  K_außen (Außeneinfl.) = {p.k_outside:.2f}
                          → {"Hoher" if p.k_outside > 0.15 else "Normaler" if p.k_outside > 0.05 else "Geringer"} Einfluss der Außentemperatur
""")
                
                print("Gemessene Werte:")
                print("─" * 50)
                print(f"""
  Starttemperatur       = {metrics['temp_start']:.1f}°C
  Endtemperatur         = {metrics['temp_end']:.1f}°C
  Temperaturänderung    = {metrics['temp_change']:+.1f}°C
  Zeit bis 63% erreicht = {time_constant:.0f} Minuten
  Zeit bis 95% erreicht = {settling_time:.0f} Minuten
""")
                
                print("Nächste Schritte:")
                print("─" * 50)
                self._print_success("Modell wurde automatisch mit neuen Parametern gespeichert!")
                self._print_instruction("Du kannst jetzt den Regelbetrieb starten:")
                print("         poetry run python main.py")
                print()
                self._print_status("Das Modell verbessert sich im Betrieb weiter (Online-Lernen)")
            
            return result
        
        except asyncio.CancelledError:
            logger.warning("Step response experiment cancelled")
            self.db.end_experiment(exp_id, {'status': 'cancelled'})
            if interactive:
                print()
                self._print_warning("Experiment wurde abgebrochen.")
                self._print_instruction("Du kannst es jederzeit neu starten.")
            raise
        
        finally:
            # Offset zurücksetzen
            if interactive:
                self._print_status("Setze Heizung zurück auf Normalbetrieb...")
            await self.ha_client.set_temperature_offset(0)
            self._running = False
            self._current_experiment = None
    
    def _find_time_constant(self, times: np.ndarray, temps: np.ndarray) -> float:
        """Findet Zeit bis 63.2% der Endänderung erreicht."""
        T_start = temps[0]
        T_end = temps[-1]
        T_63 = T_start + 0.632 * (T_end - T_start)
        
        idx = np.argmin(np.abs(temps - T_63))
        return times[idx]
    
    def _find_settling_time(self, times: np.ndarray, temps: np.ndarray, fraction: float) -> float:
        """Findet Zeit bis fraction% der Endänderung erreicht."""
        T_start = temps[0]
        T_end = temps[-1]
        T_target = T_start + fraction * (T_end - T_start)
        
        idx = np.argmin(np.abs(temps - T_target))
        return times[idx]
    
    # -------------------------------------------------------------------------
    # PRBS (Pseudo-Random Binary Sequence)
    # -------------------------------------------------------------------------
    
    async def run_prbs(
        self,
        amplitude: Optional[float] = None,
        min_duration: Optional[int] = None,
        total_duration: Optional[int] = None,
        interactive: bool = True,
    ) -> ExperimentResult:
        """
        Führt PRBS-Experiment durch.
        
        WAS PASSIERT:
        Die Heizung wechselt zufällig zwischen "mehr heizen" und "weniger heizen".
        Das ist robuster als die Sprungantwort und liefert bessere Daten
        für komplexe Modelle.
        
        DAUER: ca. 4 Stunden
        
        Args:
            amplitude: Offset-Amplitude [°C]
            min_duration: Min. Haltedauer pro Zustand [min]
            total_duration: Gesamtdauer [min]
            interactive: Wenn True, zeigt Fortschritt und Anweisungen
        """
        cfg = self.config
        amplitude = amplitude or cfg.prbs_amplitude
        min_duration = min_duration or cfg.prbs_min_duration
        total_duration = total_duration or cfg.prbs_total_duration
        
        if interactive:
            self._print_banner("🔬 SYSTEMIDENTIFIKATION: PRBS-TEST")
            
            print("Was wird gemacht:")
            print("─" * 50)
            print(f"""
PRBS = Pseudo-Random Binary Sequence

Die Heizung wechselt zufällig zwischen zwei Zuständen:
  • Mehr heizen  (Offset = -{amplitude}°C)
  • Weniger heizen (Offset = +{amplitude}°C)

Jeder Zustand wird mindestens {min_duration} Minuten gehalten.

Warum? Diese Methode ist robuster als die Sprungantwort und
liefert bessere Daten für das adaptive Modell.
""")
            
            estimated_end = datetime.now() + timedelta(minutes=total_duration)
            self._print_status(f"Geschätzte Dauer: {total_duration} Minuten ({total_duration/60:.1f} Stunden)")
            self._print_status(f"Voraussichtliches Ende: {estimated_end.strftime('%H:%M Uhr')}")
            print()
            
            self._print_warning("FENSTER MÜSSEN GESCHLOSSEN BLEIBEN!")
            print()
            
            self._print_instruction("Drücke ENTER um zu starten (oder Ctrl+C zum Abbrechen)")
            try:
                input()
            except KeyboardInterrupt:
                print("\n❌ Abgebrochen.")
                return None
        
        self._running = True
        self._current_experiment = ExperimentType.PRBS
        self._stop_event.clear()
        
        exp_id = self.db.start_experiment(
            name=f"PRBS {datetime.now().strftime('%Y%m%d_%H%M')}",
            exp_type="prbs",
            parameters={
                'amplitude': amplitude,
                'min_duration': min_duration,
                'total_duration': total_duration,
            }
        )
        
        start_time = datetime.now()
        times = []
        temps = []
        offsets = []
        outside_temps = []
        
        try:
            if interactive:
                self._print_status("PRBS-Experiment gestartet...")
                print()
            
            await self._report_progress(0, "Starting PRBS experiment")
            
            elapsed_minutes = 0
            current_offset = -amplitude  # Start mit mehr heizen
            hold_time = min_duration + np.random.randint(0, min_duration)
            hold_counter = 0
            switches = 0
            
            while elapsed_minutes < total_duration:
                if self._stop_event.is_set():
                    raise asyncio.CancelledError("Experiment stopped")
                
                data = await self._collect_sample(current_offset)
                await self._save_measurement(data, current_offset, 21.0, 'experiment')
                
                # Fenster-Check
                if data.window_open:
                    if interactive:
                        print()
                        self._print_warning("FENSTER WURDE GEÖFFNET! Experiment wird abgebrochen.")
                    raise asyncio.CancelledError("Window opened during experiment")
                
                times.append(elapsed_minutes)
                temps.append(data.room_temp)
                offsets.append(current_offset)
                outside_temps.append(data.outside_temp or 5.0)
                
                hold_counter += cfg.sample_interval / 60
                elapsed_minutes += cfg.sample_interval / 60
                
                # Offset wechseln?
                if hold_counter >= hold_time:
                    current_offset = -current_offset
                    hold_time = min_duration + np.random.randint(0, min_duration)
                    hold_counter = 0
                    switches += 1
                    logger.debug(f"PRBS: Switched to offset {current_offset}°C for {hold_time} min")
                
                if interactive:
                    progress = elapsed_minutes / total_duration
                    remaining = total_duration - int(elapsed_minutes)
                    mode = "🔥 HEIZEN" if current_offset < 0 else "❄️ PAUSE"
                    self._print_progress_bar(progress)
                    print(f"  {mode} | T={data.room_temp:.1f}°C | noch {remaining} min", end="")
                
                await asyncio.sleep(cfg.sample_interval)
            
            if interactive:
                print()
                self._print_success(f"PRBS-Experiment abgeschlossen. {switches} Zustandswechsel.")
            
            await self._report_progress(0.95, "Analyzing PRBS data")
            
            result = ExperimentResult(
                experiment_type=ExperimentType.PRBS,
                start_time=start_time,
                end_time=datetime.now(),
                times=np.array(times),
                temps=np.array(temps),
                offsets=np.array(offsets),
                outside_temps=np.array(outside_temps),
                metrics={
                    'n_samples': len(times),
                    'temp_min': np.min(temps),
                    'temp_max': np.max(temps),
                    'temp_std': np.std(temps),
                }
            )
            
            self.db.end_experiment(exp_id, result.to_dict())
            await self._report_progress(1.0, "PRBS experiment completed")
            
            return result
        
        except asyncio.CancelledError:
            logger.warning("PRBS experiment cancelled")
            self.db.end_experiment(exp_id, {'status': 'cancelled'})
            raise
        
        finally:
            await self.ha_client.set_temperature_offset(0)
            self._running = False
            self._current_experiment = None
    
    # -------------------------------------------------------------------------
    # Relay-Feedback Autotuning
    # -------------------------------------------------------------------------
    
    async def run_relay_feedback(
        self,
        target_temp: float,
        high_offset: float = -3.0,
        low_offset: float = 3.0,
        hysteresis: Optional[float] = None,
        max_duration: Optional[int] = None,
        min_oscillations: int = 3,
        interactive: bool = True,
    ) -> Dict:
        """
        Relay-Feedback Experiment für Ziegler-Nichols Autotuning.
        
        WAS PASSIERT:
        Die Heizung wechselt automatisch zwischen AN und AUS,
        wenn die Temperatur über/unter dem Sollwert liegt.
        Aus den entstehenden Schwingungen werden PID-Parameter berechnet.
        
        DAUER: ca. 2-3 Stunden (bis genug Schwingungen gemessen wurden)
        
        Returns:
            Dict mit Ku, Tu und empfohlenen PID-Parametern
        """
        cfg = self.config
        hysteresis = hysteresis or cfg.relay_hysteresis
        max_duration = max_duration or cfg.relay_max_duration
        
        # Experiment in DB registrieren
        exp_id = self.db.start_experiment(
            name=f"Relay-Feedback @ {target_temp}°C",
            exp_type="relay",
            parameters={
                'target_temp': target_temp,
                'high_offset': high_offset,
                'low_offset': low_offset,
                'hysteresis': hysteresis,
                'max_duration': max_duration,
                'min_oscillations': min_oscillations,
            }
        )
        
        if interactive:
            self._print_banner("🔬 SYSTEMIDENTIFIKATION: RELAY-FEEDBACK")
            
            print("Was wird gemacht:")
            print("─" * 50)
            print(f"""
Dieses Experiment lässt die Temperatur um den Sollwert schwingen:

  • Temperatur < {target_temp - hysteresis}°C → Heizung AN  (Offset = {high_offset}°C)
  • Temperatur > {target_temp + hysteresis}°C → Heizung AUS (Offset = {low_offset}°C)

Aus den Schwingungen werden optimale PID-Parameter berechnet
(Ziegler-Nichols Methode).

Das Experiment endet automatisch nach {min_oscillations} Schwingungen
oder spätestens nach {max_duration} Minuten.
""")
            
            self._print_status(f"Solltemperatur: {target_temp}°C")
            self._print_status(f"Maximale Dauer: {max_duration} Minuten ({max_duration/60:.1f} Stunden)")
            print()
            
            self._print_warning("FENSTER MÜSSEN GESCHLOSSEN BLEIBEN!")
            print()
            
            self._print_instruction("Drücke ENTER um zu starten (oder Ctrl+C zum Abbrechen)")
            try:
                input()
            except KeyboardInterrupt:
                print("\n❌ Abgebrochen.")
                return {'status': 'cancelled'}
        
        self._running = True
        self._current_experiment = ExperimentType.RELAY
        self._stop_event.clear()
        
        if interactive:
            self._print_status("Relay-Feedback gestartet...")
            self._print_status(f"Warte auf {min_oscillations} vollständige Schwingungen...")
            print()
        
        await self._report_progress(0, "Starting relay-feedback experiment")
        
        temps = []
        times = []
        offsets = []
        crossings = []  # Zeitpunkte der Nulldurchgänge
        
        start_time = datetime.now()
        current_offset = high_offset
        elapsed_minutes = 0
        
        try:
            while elapsed_minutes < max_duration:
                if self._stop_event.is_set():
                    raise asyncio.CancelledError()
                
                data = await self._collect_sample(current_offset)
                await self._save_measurement(data, current_offset, target_temp, 'experiment')
                
                # Fenster-Check
                if data.window_open:
                    if interactive:
                        print()
                        self._print_warning("FENSTER WURDE GEÖFFNET! Experiment wird abgebrochen.")
                    raise asyncio.CancelledError("Window opened during experiment")
                
                elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                times.append(elapsed_minutes)
                temps.append(data.room_temp)
                offsets.append(current_offset)
                
                # Relay-Logik mit Hysterese
                error = data.room_temp - target_temp
                switched = False
                
                if current_offset == high_offset and error > hysteresis:
                    current_offset = low_offset
                    crossings.append(elapsed_minutes)
                    switched = True
                    logger.debug(f"Relay: Switched to low at {elapsed_minutes:.1f} min")
                elif current_offset == low_offset and error < -hysteresis:
                    current_offset = high_offset
                    crossings.append(elapsed_minutes)
                    switched = True
                    logger.debug(f"Relay: Switched to high at {elapsed_minutes:.1f} min")
                
                if interactive:
                    oscillations = len(crossings) // 2
                    mode = "🔥 HEIZEN" if current_offset == high_offset else "❄️ PAUSE"
                    err_str = f"Δ{error:+.1f}°C"
                    
                    progress = min(oscillations / min_oscillations, elapsed_minutes / max_duration)
                    self._print_progress_bar(progress)
                    print(f"  {mode} | T={data.room_temp:.1f}°C ({err_str}) | {oscillations}/{min_oscillations} Schwingungen", end="")
                    
                    if switched:
                        print()  # Neue Zeile bei Wechsel
                
                # Genug Oszillationen?
                if len(crossings) >= min_oscillations * 2 + 1:
                    if interactive:
                        print()
                        self._print_success(f"Genug Schwingungen gemessen!")
                    break
                
                await asyncio.sleep(cfg.sample_interval)
            
            # Auswertung
            if interactive:
                self._print_banner("Auswertung", "─")
                self._print_status("Analysiere Schwingungen...")
            
            await self._report_progress(0.95, "Analyzing relay-feedback data")
            
            temps = np.array(temps)
            times = np.array(times)
            
            if len(crossings) < 4:
                logger.warning("Not enough oscillations for tuning")
                if interactive:
                    self._print_warning("Nicht genug Schwingungen für Auswertung!")
                    self._print_instruction("Versuche es erneut mit längerer Laufzeit.")
                return {'status': 'insufficient_data'}
            
            # Oszillationsperiode
            periods = np.diff(crossings)
            Tu = np.mean(periods) * 2  # Eine volle Periode = 2 Halbperioden
            
            # Amplitude
            amplitude_relay = abs(high_offset - low_offset) / 2
            amplitude_temp = (np.max(temps[-len(temps)//2:]) - np.min(temps[-len(temps)//2:])) / 2
            
            # Ultimate Gain (approximiert)
            # Ku ≈ 4 * d / (π * a) für Relay mit Amplitude d und Ausgangsamplitude a
            Ku = 4 * amplitude_relay / (np.pi * amplitude_temp) if amplitude_temp > 0.01 else 1.0
            
            # Ziegler-Nichols Tuning Rules
            pid_params = {
                'P': {'Kp': 0.5 * Ku},
                'PI': {'Kp': 0.45 * Ku, 'Ti': Tu / 1.2},
                'PID': {'Kp': 0.6 * Ku, 'Ti': Tu / 2, 'Td': Tu / 8},
                'PID_no_overshoot': {'Kp': 0.2 * Ku, 'Ti': Tu / 2, 'Td': Tu / 3},
            }
            
            result = {
                'status': 'success',
                'Ku': Ku,
                'Tu': Tu,
                'amplitude_temp': amplitude_temp,
                'n_oscillations': len(crossings) // 2,
                'pid_params': pid_params,
            }
            
            await self._report_progress(1.0, f"Relay-feedback: Ku={Ku:.2f}, Tu={Tu:.1f}min")
            
            if interactive:
                self._print_banner("🎉 ERGEBNIS", "═")
                
                print("Identifizierte Parameter:")
                print("─" * 50)
                print(f"""
  Ku (Ultimate Gain)    = {Ku:.2f}
  Tu (Schwingperiode)   = {Tu:.0f} Minuten
  Temperaturamplitude   = ±{amplitude_temp:.1f}°C
  Anzahl Schwingungen   = {len(crossings) // 2}
""")
                
                print("Empfohlene PID-Parameter (Ziegler-Nichols):")
                print("─" * 50)
                print(f"""
  PID (klassisch):
    Kp = {pid_params['PID']['Kp']:.2f}
    Ti = {pid_params['PID']['Ti']:.0f} min
    Td = {pid_params['PID']['Td']:.0f} min
    
  PID (ohne Überschwingen):
    Kp = {pid_params['PID_no_overshoot']['Kp']:.2f}
    Ti = {pid_params['PID_no_overshoot']['Ti']:.0f} min
    Td = {pid_params['PID_no_overshoot']['Td']:.0f} min
""")
                
                self._print_status("Diese Werte sind für einen klassischen PID-Regler.")
                self._print_status("Der MPC-Controller verwendet das adaptive Modell.")
            
            # Ergebnis in DB speichern
            self.db.end_experiment(exp_id, result)
            
            return result
        
        except asyncio.CancelledError:
            logger.warning("Relay-feedback cancelled")
            if interactive:
                print()
                self._print_warning("Experiment wurde abgebrochen.")
            self.db.end_experiment(exp_id, {'status': 'cancelled'})
            raise
        
        finally:
            if interactive:
                self._print_status("Setze Heizung zurück auf Normalbetrieb...")
            await self.ha_client.set_temperature_offset(0)
            self._running = False
            self._current_experiment = None
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def current_experiment(self) -> Optional[ExperimentType]:
        return self._current_experiment


if __name__ == "__main__":
    # Nur für Struktur-Test
    print("ExperimentRunner module loaded")
