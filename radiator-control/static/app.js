// Radiator Control Add-on JavaScript v1.1

const API_BASE = '';

// Status polling
let statusInterval = null;
let statsInterval = null;
let graphData = { times: [], room: [], target: [], outside: [] };

// =============================================================================
// API Functions
// =============================================================================

async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(API_BASE + endpoint, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        return null;
    }
}

// =============================================================================
// Status Updates
// =============================================================================

async function updateStatus() {
    const data = await fetchAPI('/api/status');
    if (!data) return;

    // Badge
    const badge = document.getElementById('status-badge');
    badge.textContent = formatMode(data.mode);
    badge.className = 'badge ' + data.mode;

    // Temperature cards
    if (data.room_temp !== null) {
        document.getElementById('room-temp').textContent = data.room_temp.toFixed(1) + '°C';
        addGraphPoint('room', data.room_temp);
    }
    if (data.target_temp !== null) {
        document.getElementById('target-temp').textContent = data.target_temp.toFixed(1) + '°C';
        addGraphPoint('target', data.target_temp);
    }
    if (data.outside_temp !== null) {
        document.getElementById('outside-temp').textContent = data.outside_temp.toFixed(1) + '°C';
        addGraphPoint('outside', data.outside_temp);
    }
    
    // Offset
    const offsetEl = document.getElementById('offset');
    offsetEl.textContent = (data.offset >= 0 ? '+' : '') + data.offset.toFixed(2) + '°C';
    offsetEl.style.color = data.offset < 0 ? '#4caf50' : (data.offset > 0 ? '#f44336' : '#fff');

    // HVAC Mode
    const hvacEl = document.getElementById('hvac-mode');
    hvacEl.textContent = formatHvacMode(data.hvac_mode);
    document.getElementById('card-hvac-mode').className = 'card ' + (data.hvac_mode === 'heat' ? 'heating' : '');

    // Window
    const windowEl = document.getElementById('window-state');
    windowEl.textContent = data.window_open ? 'Offen' : 'Geschlossen';
    document.getElementById('card-window').className = 'card ' + (data.window_open ? 'window-open' : '');

    // Info Section
    document.getElementById('info-mode').textContent = formatMode(data.mode);
    document.getElementById('info-hvac').textContent = formatHvacMode(data.hvac_mode);
    document.getElementById('info-heating').textContent = data.heating_active ? '🔥 Ja' : '❄️ Nein';
    document.getElementById('info-window').textContent = data.window_open ? '🪟 Offen' : '✅ Geschlossen';
    document.getElementById('info-update').textContent = data.last_update ? formatTime(data.last_update) : '--';

    // Buttons
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    
    if (data.running || data.mode === 'experiment') {
        btnStart.disabled = true;
        btnStop.disabled = false;
    } else {
        btnStart.disabled = false;
        btnStop.disabled = true;
    }

    // Experiment progress
    if (data.experiment) {
        document.getElementById('experiment-progress').style.display = 'block';
        document.getElementById('experiment-name').textContent = 
            `Experiment: ${formatExperimentType(data.experiment)}`;
        document.getElementById('progress-fill').style.width = 
            (data.experiment_progress * 100) + '%';
        document.getElementById('progress-text').textContent = 
            Math.round(data.experiment_progress * 100) + '%';
        
        // Disable experiment buttons
        document.querySelectorAll('.experiment-card button').forEach(btn => btn.disabled = true);
    } else {
        document.getElementById('experiment-progress').style.display = 'none';
        document.querySelectorAll('.experiment-card button').forEach(btn => btn.disabled = data.running);
    }

    // Entities
    if (data.entities) {
        document.getElementById('info-entity-thermo').textContent = data.entities.thermostat || '--';
        document.getElementById('info-entity-temp').textContent = data.entities.temp_sensor || '--';
        document.getElementById('info-entity-window').textContent = data.entities.window_sensor || '--';
        document.getElementById('info-entity-outside').textContent = data.entities.outside_temp || '--';
    }

    // Last update time
    document.getElementById('last-update-time').textContent = new Date().toLocaleTimeString('de-DE');
    
    // Update graph
    drawGraph();
}

async function updateStats() {
    const data = await fetchAPI('/api/stats');
    if (!data) return;

    const summary = data.summary;
    
    if (summary && summary.status !== 'no_data') {
        // Performance info
        if (summary.current_rmse !== null) {
            document.getElementById('info-perf-rmse').textContent = summary.current_rmse.toFixed(3) + ' °C';
        }
        if (summary.current_mae !== null) {
            document.getElementById('info-perf-mae').textContent = summary.current_mae.toFixed(3) + ' °C';
        }
        if (summary.current_comfort !== null) {
            document.getElementById('info-perf-comfort').textContent = summary.current_comfort.toFixed(1) + '%';
        }
        if (summary.trend) {
            document.getElementById('info-perf-trend').textContent = formatTrend(summary.trend);
        }
        if (summary.total_samples) {
            document.getElementById('info-perf-samples').textContent = summary.total_samples.toLocaleString();
        }
    }

    // Daily table
    const tbody = document.querySelector('#daily-stats tbody');
    if (data.daily && data.daily.length > 0) {
        tbody.innerHTML = data.daily.map(row => `
            <tr>
                <td>${formatDate(row.date)}</td>
                <td class="${getRMSEClass(row.rmse)}">${row.rmse.toFixed(3)}°C</td>
                <td>${row.mae.toFixed(3)}°C</td>
                <td>${row.comfort_percent.toFixed(1)}%</td>
                <td>${row.samples}</td>
                <td>${row.avg_offset >= 0 ? '+' : ''}${row.avg_offset.toFixed(2)}°C</td>
            </tr>
        `).join('');
    } else {
        tbody.innerHTML = '<tr><td colspan="6">Noch keine Daten</td></tr>';
    }
}

async function updateModel() {
    const data = await fetchAPI('/api/model');
    if (!data) return;

    if (data.params) {
        document.getElementById('info-tau').textContent = data.params.tau.toFixed(1) + ' min';
        document.getElementById('info-k-heater').textContent = data.params.k_heater.toFixed(3);
        document.getElementById('info-k-outside').textContent = (data.params.k_outside || 0).toFixed(4);
    }
    
    document.getElementById('info-model-rmse').textContent = data.rmse.toFixed(4) + ' °C';
    document.getElementById('info-updates').textContent = data.n_updates.toLocaleString();
}

async function updateExperiments() {
    const data = await fetchAPI('/api/experiments');
    if (!data) return;

    const container = document.getElementById('experiments-list');
    
    if (data.experiments && data.experiments.length > 0) {
        container.innerHTML = data.experiments.slice(0, 10).map(exp => `
            <div class="experiment-item ${exp.status}">
                <div class="exp-info">
                    <strong>${exp.name}</strong>
                    <span class="exp-time">${formatDateTime(exp.start_time)} • ${exp.duration_minutes ? Math.round(exp.duration_minutes) + ' min' : '?'}</span>
                </div>
                <span class="status ${exp.status}">${formatExpStatus(exp.status)}</span>
            </div>
        `).join('');
    } else {
        container.innerHTML = '<p class="no-data">Noch keine Experimente durchgeführt</p>';
    }
}

async function updateConfig() {
    const data = await fetchAPI('/api/config');
    if (!data) return;
    
    // Populate settings fields
    if (data.mpc) {
        document.getElementById('set-horizon').value = data.mpc.horizon_minutes || 120;
        document.getElementById('set-weight-comfort').value = data.mpc.weight_comfort || 1.0;
        document.getElementById('set-weight-energy').value = data.mpc.weight_energy || 0.1;
        document.getElementById('set-weight-smooth').value = data.mpc.weight_smoothness || 0.05;
    }
    if (data.control) {
        document.getElementById('set-offset-min').value = data.control.offset_min || -5;
        document.getElementById('set-offset-max').value = data.control.offset_max || 5;
        document.getElementById('set-window-action').value = data.control.window_action || 'turn_off';
        document.getElementById('set-window-delay').value = data.control.window_off_delay || 30;
    }
    if (data.model) {
        document.getElementById('set-forgetting').value = data.model.forgetting_factor || 0.98;
    }
}

// =============================================================================
// Control Functions
// =============================================================================

async function startControl() {
    const result = await fetchAPI('/api/control/start', { method: 'POST' });
    if (result && result.status === 'started') {
        showNotification('Regelung gestartet', 'success');
    }
    updateStatus();
}

async function stopControl() {
    const result = await fetchAPI('/api/control/stop', { method: 'POST' });
    if (result) {
        showNotification('Regelung wird gestoppt...', 'info');
    }
    updateStatus();
}

async function turnOnThermostat() {
    const result = await fetchAPI('/api/thermostat/on', { method: 'POST' });
    if (result) {
        showNotification('Thermostat eingeschaltet', 'success');
    }
    updateStatus();
}

async function turnOffThermostat() {
    const result = await fetchAPI('/api/thermostat/off', { method: 'POST' });
    if (result) {
        showNotification('Thermostat ausgeschaltet', 'info');
    }
    updateStatus();
}

async function startExperiment(type) {
    if (!confirm(`Experiment "${formatExperimentType(type)}" starten?\n\nWichtig:\n- Fenster geschlossen halten!\n- Raum möglichst nicht betreten\n- Kann mehrere Stunden dauern`)) {
        return;
    }
    
    const result = await fetchAPI('/api/experiment/start', {
        method: 'POST',
        body: JSON.stringify({ type }),
    });
    
    if (result && result.status === 'started') {
        showNotification(`Experiment ${formatExperimentType(type)} gestartet`, 'success');
    }
    updateStatus();
}

async function stopExperiment() {
    if (!confirm('Experiment wirklich abbrechen?\n\nDie bisherigen Daten gehen verloren.')) return;
    
    const result = await fetchAPI('/api/experiment/stop', { method: 'POST' });
    if (result) {
        showNotification('Experiment abgebrochen', 'warning');
    }
    updateStatus();
}

async function saveSettings() {
    const settings = {
        mpc: {
            horizon_minutes: parseInt(document.getElementById('set-horizon').value),
            weight_comfort: parseFloat(document.getElementById('set-weight-comfort').value),
            weight_energy: parseFloat(document.getElementById('set-weight-energy').value),
            weight_smoothness: parseFloat(document.getElementById('set-weight-smooth').value),
        },
        control: {
            offset_min: parseFloat(document.getElementById('set-offset-min').value),
            offset_max: parseFloat(document.getElementById('set-offset-max').value),
            window_action: document.getElementById('set-window-action').value,
            window_off_delay: parseInt(document.getElementById('set-window-delay').value),
        },
        model: {
            forgetting_factor: parseFloat(document.getElementById('set-forgetting').value),
        },
    };
    
    const result = await fetchAPI('/api/config', {
        method: 'POST',
        body: JSON.stringify(settings),
    });
    
    if (result && result.status === 'saved') {
        showNotification('Einstellungen gespeichert', 'success');
        toggleSettings();
    } else {
        showNotification('Fehler beim Speichern', 'error');
    }
}

async function resetModel() {
    if (!confirm('Modell wirklich zurücksetzen?\n\nAlle gelernten Parameter werden gelöscht!')) return;
    
    const result = await fetchAPI('/api/model/reset', { method: 'POST' });
    if (result) {
        showNotification('Modell zurückgesetzt', 'warning');
        updateModel();
    }
}

function toggleSettings() {
    const panel = document.getElementById('settings-panel');
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    if (panel.style.display === 'block') {
        updateConfig();
    }
}

function copyEntityId() {
    const entityId = document.getElementById('climate-entity-id').textContent;
    navigator.clipboard.writeText(entityId);
    showNotification('Entity ID kopiert', 'success');
}

// =============================================================================
// Graph
// =============================================================================

function addGraphPoint(series, value) {
    const now = new Date();
    
    // Initialize if needed
    if (graphData.times.length === 0) {
        graphData.times.push(now);
        graphData.room.push(series === 'room' ? value : null);
        graphData.target.push(series === 'target' ? value : null);
        graphData.outside.push(series === 'outside' ? value : null);
    } else {
        const lastTime = graphData.times[graphData.times.length - 1];
        
        // Only add new point if enough time has passed (1 min)
        if (now - lastTime > 60000) {
            graphData.times.push(now);
            graphData.room.push(null);
            graphData.target.push(null);
            graphData.outside.push(null);
        }
        
        // Update latest value
        const idx = graphData[series].length - 1;
        graphData[series][idx] = value;
    }
    
    // Keep only last 24 hours
    const cutoff = new Date(now - 24 * 60 * 60 * 1000);
    while (graphData.times.length > 0 && graphData.times[0] < cutoff) {
        graphData.times.shift();
        graphData.room.shift();
        graphData.target.shift();
        graphData.outside.shift();
    }
}

function drawGraph() {
    const canvas = document.getElementById('temp-graph');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.parentElement.clientWidth;
    const height = 200;
    
    canvas.width = width;
    canvas.height = height;
    
    // Clear
    ctx.fillStyle = '#1c1c1c';
    ctx.fillRect(0, 0, width, height);
    
    if (graphData.times.length < 2) {
        ctx.fillStyle = '#666';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Warte auf Daten...', width / 2, height / 2);
        return;
    }
    
    // Find min/max
    const allValues = [...graphData.room, ...graphData.target, ...graphData.outside].filter(v => v !== null);
    const minVal = Math.floor(Math.min(...allValues) - 1);
    const maxVal = Math.ceil(Math.max(...allValues) + 1);
    
    const padding = { top: 20, right: 20, bottom: 30, left: 40 };
    const graphWidth = width - padding.left - padding.right;
    const graphHeight = height - padding.top - padding.bottom;
    
    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    
    for (let v = minVal; v <= maxVal; v++) {
        const y = padding.top + graphHeight - ((v - minVal) / (maxVal - minVal)) * graphHeight;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        
        ctx.fillStyle = '#666';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(v + '°', padding.left - 5, y + 4);
    }
    
    // Draw lines
    const timeRange = graphData.times[graphData.times.length - 1] - graphData.times[0];
    
    function drawLine(data, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        let started = false;
        for (let i = 0; i < data.length; i++) {
            if (data[i] === null) continue;
            
            const x = padding.left + ((graphData.times[i] - graphData.times[0]) / timeRange) * graphWidth;
            const y = padding.top + graphHeight - ((data[i] - minVal) / (maxVal - minVal)) * graphHeight;
            
            if (!started) {
                ctx.moveTo(x, y);
                started = true;
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }
    
    drawLine(graphData.outside, '#03a9f4');
    drawLine(graphData.target, '#ff9800');
    drawLine(graphData.room, '#4caf50');
}

// =============================================================================
// Helper Functions
// =============================================================================

function formatMode(mode) {
    const modes = {
        'idle': 'Bereit',
        'control': 'Regelung aktiv',
        'experiment': 'Experiment',
        'window_open': 'Fenster offen',
        'stopping': 'Wird gestoppt...',
    };
    return modes[mode] || mode;
}

function formatHvacMode(mode) {
    const modes = {
        'heat': '🔥 Heizen',
        'off': '❄️ Aus',
        'auto': '🔄 Auto',
        'cool': '❄️ Kühlen',
    };
    return modes[mode] || mode || '--';
}

function formatExperimentType(type) {
    const types = {
        'step': 'Sprungantwort',
        'prbs': 'PRBS-Test',
        'relay': 'Relay-Feedback',
    };
    return types[type] || type;
}

function formatExpStatus(status) {
    const statuses = {
        'completed': '✅ Abgeschlossen',
        'cancelled': '❌ Abgebrochen',
        'running': '⏳ Läuft...',
    };
    return statuses[status] || status;
}

function formatTrend(trend) {
    const trends = {
        'improving': '📈 Besser',
        'worsening': '📉 Schlechter',
        'stable': '➡️ Stabil',
    };
    return trends[trend] || trend;
}

function formatDate(dateStr) {
    if (!dateStr) return '?';
    const date = new Date(dateStr);
    return date.toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit' });
}

function formatDateTime(dateStr) {
    if (!dateStr) return '?';
    const date = new Date(dateStr);
    return date.toLocaleDateString('de-DE') + ' ' + 
           date.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
}

function formatTime(dateStr) {
    if (!dateStr) return '?';
    const date = new Date(dateStr);
    return date.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function getRMSEClass(rmse) {
    if (rmse < 0.3) return 'good';
    if (rmse < 0.5) return '';
    if (rmse < 0.8) return 'warning';
    return 'bad';
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notif = document.createElement('div');
    notif.className = `notification ${type}`;
    notif.textContent = message;
    document.body.appendChild(notif);
    
    // Animate in
    setTimeout(() => notif.classList.add('show'), 10);
    
    // Remove after 3s
    setTimeout(() => {
        notif.classList.remove('show');
        setTimeout(() => notif.remove(), 300);
    }, 3000);
}

// =============================================================================
// Initialization
// =============================================================================

function init() {
    // Initial load
    updateStatus();
    updateStats();
    updateModel();
    updateExperiments();
    
    // Polling
    statusInterval = setInterval(updateStatus, 2000);
    statsInterval = setInterval(() => {
        updateStats();
        updateModel();
        updateExperiments();
    }, 30000);
    
    // Resize handler for graph
    window.addEventListener('resize', drawGraph);
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
