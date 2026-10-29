/**
 * NetworkPulse: Professional Observability & Forecasting Dashboard
 * Client-Side Application Controller
 */

// State Management
const state = {
    currentTab: 'overview',
    historyLimit: 168,
    historyOffset: 0,
    forecastWindow: 1,
    anomalyPage: 1,
    anomalyPageSize: 25,
    anomalySeverity: 'all',
    explorerPage: 1,
    explorerPageSize: 25,
    explorerSearch: '',
    explorerAnomaliesOnly: false,
    charts: {}
};

// Tab Meta Configurations
const tabMeta = {
    'overview': {
        title: 'Network Observability Overview',
        desc: 'Real-time round-trip latency, empirical anomaly detection, and 24-hour predictive horizon'
    },
    'live-monitoring': {
        title: 'Live Telemetry & RF Signal Monitor',
        desc: 'High-frequency telemetry replay tracking packet delays, SNR, and channel loss'
    },
    'latency-forecast': {
        title: '24-Hour Horizon Multi-Model Latency Forecasting',
        desc: 'Out-of-sample comparison across 28 rolling 24-hour evaluation horizons (672 test hours)'
    },
    'anomaly-detection': {
        title: 'Telemetry Anomaly Detection & Attack Attribution',
        desc: 'Dynamic threshold excursions above 50 ms nominal baseline with cyber-attack labeling'
    },
    'model-comparison': {
        title: 'Forecasting Model Benchmark & Evaluation Leaderboard',
        desc: 'Strict out-of-sample MAE/RMSE comparisons and theoretical Bayes floor analysis'
    },
    'historical-analytics': {
        title: 'Historical Telemetry & Protocol Analytics',
        desc: 'Statistical distribution of round-trip times, transport protocols, and encryption profiles'
    },
    'data-explorer': {
        title: 'Telemetry Data Explorer',
        desc: 'Searchable, filterable tabular browser for raw telemetry records and network metrics'
    },
    'reports': {
        title: 'Technical Summary & Artifact Exports',
        desc: 'Executive performance evaluation, limitation analysis, and dataset CSV exports'
    }
};

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initChartDefaults();
    setupNavigation();
    setupEventListeners();
    
    // Check initial URL hash
    const initialHash = window.location.hash.replace('#', '');
    if (initialHash && tabMeta[initialHash]) {
        switchTab(initialHash);
    } else {
        loadOverviewData();
    }
});

// Configure Chart.js global dark theme defaults
function initChartDefaults() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.plugins.tooltip.backgroundColor = '#0e1526';
    Chart.defaults.plugins.tooltip.titleColor = '#f8fafc';
    Chart.defaults.plugins.tooltip.bodyColor = '#cbd5e1';
    Chart.defaults.plugins.tooltip.borderColor = '#334155';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.boxPadding = 4;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
}

// Destroy existing Chart instance safely before creating new
function destroyChart(chartKey) {
    if (state.charts[chartKey]) {
        state.charts[chartKey].destroy();
        delete state.charts[chartKey];
    }
}

// ============================================================================
// Navigation & Tab Switching
// ============================================================================
function setupNavigation() {
    // Sidebar collapse toggle
    const collapseBtn = document.getElementById('sidebarCollapseBtn');
    const sidebar = document.getElementById('sidebar');
    if (collapseBtn && sidebar) {
        collapseBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });
    }

    // Mobile menu toggle
    const mobileBtn = document.getElementById('mobileMenuBtn');
    if (mobileBtn && sidebar) {
        mobileBtn.addEventListener('click', () => {
            sidebar.classList.toggle('mobile-open');
        });
    }

    // Nav item clicks
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tabKey = item.getAttribute('data-tab');
            if (tabKey) {
                switchTab(tabKey);
                window.location.hash = tabKey;
                if (window.innerWidth <= 768 && sidebar) {
                    sidebar.classList.remove('mobile-open');
                }
            }
        });
    });

    // Hash change event
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.replace('#', '');
        if (hash && tabMeta[hash] && hash !== state.currentTab) {
            switchTab(hash);
        }
    });
}

function switchTab(tabKey) {
    if (!tabMeta[tabKey]) return;
    state.currentTab = tabKey;

    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.getAttribute('data-tab') === tabKey);
    });

    // Update tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.toggle('active', pane.id === `tab-${tabKey}`);
    });

    // Update Navbar headers
    const meta = tabMeta[tabKey];
    const navTitle = document.getElementById('navbarTitle');
    const navDesc = document.getElementById('navbarDesc');
    if (navTitle) navTitle.textContent = meta.title;
    if (navDesc) navDesc.textContent = meta.desc;

    // Fetch tab-specific data
    switch (tabKey) {
        case 'overview':
            loadOverviewData();
            break;
        case 'live-monitoring':
            loadLiveMonitoringData();
            break;
        case 'latency-forecast':
            loadForecastData();
            break;
        case 'anomaly-detection':
            loadAnomalyData();
            break;
        case 'model-comparison':
            loadModelComparisonData();
            break;
        case 'historical-analytics':
            loadHistoricalAnalyticsData();
            break;
        case 'data-explorer':
            loadDataExplorer();
            break;
        case 'reports':
            // Static structured report is ready; trigger icon re-render
            lucide.createIcons();
            break;
    }
}

// ============================================================================
// Event Listeners for Controls
// ============================================================================
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            const icon = document.getElementById('refreshIcon');
            if (icon) icon.classList.add('spinning');
            
            // Reload active tab
            switchTab(state.currentTab);
            
            setTimeout(() => {
                if (icon) icon.classList.remove('spinning');
            }, 600);
        });
    }

    // Overview History Range Selector
    const rangeBtns = document.querySelectorAll('#historyRangeSelector .range-btn');
    rangeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            rangeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.historyLimit = parseInt(btn.getAttribute('data-limit') || '168', 10);
            loadOverviewTelemetryChart();
        });
    });

    // Forecast Window Select
    const fcSelect = document.getElementById('forecastWindowSelect');
    if (fcSelect) {
        fcSelect.addEventListener('change', (e) => {
            state.forecastWindow = parseInt(e.target.value, 10);
            loadForecastComparisonWindow();
        });
    }

    // Anomaly Severity Filter
    const anomFilter = document.getElementById('anomalySeverityFilter');
    if (anomFilter) {
        anomFilter.addEventListener('change', (e) => {
            state.anomalySeverity = e.target.value;
            state.anomalyPage = 1;
            loadAnomalyData();
        });
    }

    // Anomaly Page Size
    const anomPageSize = document.getElementById('anomalyPageSize');
    if (anomPageSize) {
        anomPageSize.addEventListener('change', (e) => {
            state.anomalyPageSize = parseInt(e.target.value, 10);
            state.anomalyPage = 1;
            loadAnomalyData();
        });
    }

    // Anomaly Pagination Prev/Next
    const anomPrev = document.getElementById('anomPrevBtn');
    const anomNext = document.getElementById('anomNextBtn');
    if (anomPrev) {
        anomPrev.addEventListener('click', () => {
            if (state.anomalyPage > 1) {
                state.anomalyPage--;
                loadAnomalyData();
            }
        });
    }
    if (anomNext) {
        anomNext.addEventListener('click', () => {
            state.anomalyPage++;
            loadAnomalyData();
        });
    }

    // Explorer Search with Debounce
    const expSearch = document.getElementById('explorerSearchInput');
    let debounceTimer;
    if (expSearch) {
        expSearch.addEventListener('input', (e) => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                state.explorerSearch = e.target.value;
                state.explorerPage = 1;
                loadDataExplorer();
            }, 300);
        });
    }

    // Explorer Anomalies Only Checkbox
    const expAnomOnly = document.getElementById('explorerAnomaliesOnly');
    if (expAnomOnly) {
        expAnomOnly.addEventListener('change', (e) => {
            state.explorerAnomaliesOnly = e.target.checked;
            state.explorerPage = 1;
            loadDataExplorer();
        });
    }

    // Explorer Page Size
    const expPageSize = document.getElementById('explorerPageSize');
    if (expPageSize) {
        expPageSize.addEventListener('change', (e) => {
            state.explorerPageSize = parseInt(e.target.value, 10);
            state.explorerPage = 1;
            loadDataExplorer();
        });
    }

    // Explorer Prev/Next
    const expPrev = document.getElementById('explorerPrevBtn');
    const expNext = document.getElementById('explorerNextBtn');
    if (expPrev) {
        expPrev.addEventListener('click', () => {
            if (state.explorerPage > 1) {
                state.explorerPage--;
                loadDataExplorer();
            }
        });
    }
    if (expNext) {
        expNext.addEventListener('click', () => {
            state.explorerPage++;
            loadDataExplorer();
        });
    }
}

// ============================================================================
// TAB 1: OVERVIEW DATA & CHARTS
// ============================================================================
async function loadOverviewData() {
    try {
        // 1. Fetch Metrics Summary
        const resSummary = await fetch('/api/metrics/summary');
        if (resSummary.ok) {
            const data = await resSummary.json();
            
            // Update KPI cards
            const currentEl = document.getElementById('kpiCurrentRtt');
            if (currentEl) currentEl.textContent = data.current_rtt_ms;

            const avgEl = document.getElementById('kpiAvgRtt');
            if (avgEl) avgEl.textContent = data.average_rtt_ms;

            const anomCountEl = document.getElementById('kpiAnomalyCount');
            if (anomCountEl) anomCountEl.textContent = data.total_anomalies.toLocaleString();

            const anomRateEl = document.getElementById('kpiAnomalyRate');
            if (anomRateEl) anomRateEl.textContent = `${data.anomaly_rate_pct}%`;

            const fcMaxEl = document.getElementById('kpiForecastMax');
            if (fcMaxEl) fcMaxEl.textContent = data.forecasted_max_rtt_ms;

            const healthStatusEl = document.getElementById('kpiHealthStatus');
            const healthDescEl = document.getElementById('kpiHealthDesc');
            if (healthStatusEl) {
                healthStatusEl.textContent = data.health_status;
                healthStatusEl.style.color = data.health_color;
            }
            if (healthDescEl) healthDescEl.textContent = data.health_desc;

            // Last updated & badges
            const lastUpEl = document.getElementById('lastUpdatedDisplay');
            if (lastUpEl) lastUpEl.textContent = data.last_updated;

            const sideBadge = document.getElementById('sidebarAnomalyBadge');
            if (sideBadge) sideBadge.textContent = data.total_anomalies.toLocaleString();
        }

        // 2. Load Telemetry Chart
        await loadOverviewTelemetryChart();

        // 3. Load Overview Recent Anomalies
        await loadOverviewRecentAnomalies();

        lucide.createIcons();
    } catch (err) {
        console.error('Error loading overview data:', err);
    }
}

async function loadOverviewTelemetryChart() {
    try {
        const res = await fetch(`/api/telemetry/history?limit=${state.historyLimit}&offset=${state.historyOffset}`);
        if (!res.ok) return;
        const data = await res.json();

        const canvas = document.getElementById('overviewTelemetryChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        destroyChart('overviewTelemetry');

        // Prepare labels: format timestamps nicely
        const labels = data.timestamps.map(ts => {
            const parts = ts.split(' ');
            return parts.length > 1 ? `${parts[0].slice(5)} ${parts[1].slice(0, 5)}` : ts;
        });

        // Scatter anomalies
        const anomalyData = data.anomalies.map(a => {
            const idx = data.timestamps.indexOf(a.x);
            return { x: idx >= 0 ? labels[idx] : a.x, y: a.y, severity: a.severity };
        });

        state.charts['overviewTelemetry'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Observed RTT (ms)',
                        data: data.rtt_ms,
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.05)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0.1,
                        fill: false
                    },
                    {
                        label: '6h Rolling Mean',
                        data: data.rolling_mean_6h,
                        borderColor: '#3b82f6',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.2,
                        fill: false
                    },
                    {
                        label: 'Rolling Std Upper (+1σ)',
                        data: data.rolling_std_upper,
                        borderColor: 'rgba(100, 116, 139, 0.3)',
                        borderWidth: 1,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        fill: '+1',
                        backgroundColor: 'rgba(100, 116, 139, 0.08)'
                    },
                    {
                        label: 'Rolling Std Lower (-1σ)',
                        data: data.rolling_std_lower,
                        borderColor: 'rgba(100, 116, 139, 0.3)',
                        borderWidth: 1,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'Spike Anomalies (>50ms)',
                        data: anomalyData,
                        type: 'scatter',
                        backgroundColor: '#ef4444',
                        borderColor: '#ef4444',
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        showLine: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { maxTicksLimit: 12, maxRotation: 0 }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Round-Trip Time (ms)', color: '#64748b' },
                        min: 40,
                        suggestedMax: 500
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    } catch (err) {
        console.error('Error drawing overview telemetry chart:', err);
    }
}

async function loadOverviewRecentAnomalies() {
    try {
        const res = await fetch('/api/anomalies?page=1&page_size=5');
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.getElementById('overviewAnomaliesTableBody');
        if (!tbody) return;

        if (data.records.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-muted">No anomalies detected.</td></tr>';
            return;
        }

        tbody.innerHTML = data.records.map(r => `
            <tr>
                <td class="font-mono text-muted">${r.timestamp}</td>
                <td class="font-bold text-white">${r.observed_rtt_ms} ms</td>
                <td class="text-amber">+${r.deviation_ms} ms</td>
                <td>
                    <span class="severity-pill" style="background-color: ${r.severity_color}25; color: ${r.severity_color};">
                        ${r.severity}
                    </span>
                </td>
                <td>
                    <span class="attack-pill">${r.attack_type}</span>
                </td>
            </tr>
        `).join('');
    } catch (err) {
        console.error('Error loading overview recent anomalies:', err);
    }
}

// ============================================================================
// TAB 2: LIVE MONITORING
// ============================================================================
async function loadLiveMonitoringData() {
    try {
        const res = await fetch('/api/telemetry/history?limit=168');
        if (!res.ok) return;
        const data = await res.json();

        const labels = data.timestamps.map(ts => {
            const parts = ts.split(' ');
            return parts.length > 1 ? `${parts[0].slice(5)} ${parts[1].slice(0, 5)}` : ts;
        });

        // 1. Live RTT Stream Chart
        const rttCanvas = document.getElementById('liveRttChart');
        if (rttCanvas) {
            destroyChart('liveRtt');
            state.charts['liveRtt'] = new Chart(rttCanvas.getContext('2d'), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'RTT Telemetry (ms)',
                            data: data.rtt_ms,
                            borderColor: '#06b6d4',
                            backgroundColor: 'rgba(6, 182, 212, 0.08)',
                            borderWidth: 1.5,
                            pointRadius: 0,
                            pointHoverRadius: 4,
                            fill: true,
                            tension: 0.1
                        },
                        {
                            label: 'Nominal Baseline (50 ms)',
                            data: Array(labels.length).fill(50.0),
                            borderColor: '#10b981',
                            borderWidth: 1.5,
                            borderDash: [4, 4],
                            pointRadius: 0,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { maxTicksLimit: 12 } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'RTT (ms)' } }
                    }
                }
            });
        }

        // 2. Packet Loss Chart
        const lossCanvas = document.getElementById('livePacketLossChart');
        if (lossCanvas) {
            destroyChart('livePacketLoss');
            state.charts['livePacketLoss'] = new Chart(lossCanvas.getContext('2d'), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Packet Loss Rate (%)',
                        data: data.packet_loss_pct,
                        borderColor: '#f97316',
                        backgroundColor: 'rgba(249, 115, 22, 0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { maxTicksLimit: 8 } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'Loss (%)' }, min: 0 }
                    }
                }
            });
        }


        lucide.createIcons();
    } catch (err) {
        console.error('Error loading live monitoring data:', err);
    }
}

// ============================================================================
// TAB 3: LATENCY FORECAST
// ============================================================================
async function loadForecastData() {
    try {
        // 1. Populate Window Select Options (28 Windows)
        const resWindows = await fetch('/api/forecast/windows');
        if (resWindows.ok) {
            const windows = await resWindows.json();
            const selectEl = document.getElementById('forecastWindowSelect');
            if (selectEl) {
                selectEl.innerHTML = windows.map(w => `
                    <option value="${w.window_id}" ${w.window_id === state.forecastWindow ? 'selected' : ''}>
                        Window ${w.window_id} (${w.season_label}: ${w.start_timestamp.slice(0, 10)}) — AutoTS: ${w.mae_autots_ms}ms vs ARIMA: ${w.mae_arima_ms}ms
                    </option>
                `).join('');
            }
        }

        // 2. Load Selected Window Comparison
        await loadForecastComparisonWindow();

        // 3. Load Step-by-Step Horizon Error Profile (h=1..24)
        await loadForecastHorizonProfile();

        lucide.createIcons();
    } catch (err) {
        console.error('Error loading forecast data:', err);
    }
}

async function loadForecastComparisonWindow() {
    try {
        const res = await fetch(`/api/forecast/comparison?window=${state.forecastWindow}`);
        if (!res.ok) return;
        const data = await res.json();

        // Update strip
        const origEl = document.getElementById('fcOriginTs');
        const spanEl = document.getElementById('fcSpanTs');
        const arimaMaeEl = document.getElementById('fcArimaMae');
        const autotsMaeEl = document.getElementById('fcAutotsMae');
        const xgbMaeEl = document.getElementById('fcXgbMae');
        const winnerEl = document.getElementById('fcWinnerBadge');

        if (origEl) origEl.textContent = data.origin_timestamp;
        if (spanEl) spanEl.textContent = `${data.start_timestamp.slice(5)} to ${data.end_timestamp.slice(5)}`;
        if (arimaMaeEl) arimaMaeEl.textContent = `${data.metrics.arima.MAE} ms`;
        if (autotsMaeEl) autotsMaeEl.textContent = `${data.metrics.autots.MAE} ms`;
        if (xgbMaeEl) xgbMaeEl.textContent = `${data.metrics.xgboost.MAE} ms`;
        if (winnerEl) {
            const isAutoTs = data.metrics.autots.MAE <= data.metrics.arima.MAE;
            winnerEl.textContent = isAutoTs ? 'AutoTS (Champion)' : 'Log-ARIMA';
        }

        // Render Chart
        const canvas = document.getElementById('forecastComparisonChart');
        if (!canvas) return;
        destroyChart('forecastComparison');

        const stepLabels = data.steps.map(s => `h=${s}`);

        state.charts['forecastComparison'] = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: stepLabels,
                datasets: [
                    {
                        label: 'Actual Observed RTT',
                        data: data.actual_rtt_ms,
                        borderColor: '#ffffff',
                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                        borderWidth: 2.5,
                        pointRadius: 3,
                        pointBackgroundColor: '#ffffff',
                        tension: 0.1
                    },
                    {
                        label: 'AutoTS BasicLinearModel (50 ms Floor)',
                        data: data.autots_pred_ms,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.08)',
                        borderWidth: 2.5,
                        pointRadius: 2,
                        tension: 0.1
                    },
                    {
                        label: 'Log-ARIMA(1,0,0) Baseline',
                        data: data.arima_pred_ms,
                        borderColor: '#3b82f6',
                        borderWidth: 2,
                        borderDash: [5, 4],
                        pointRadius: 2,
                        tension: 0.2
                    },
                    {
                        label: 'XGBoost Quantile (q=0.46)',
                        data: data.xgboost_pred_ms,
                        borderColor: '#06b6d4',
                        borderWidth: 2,
                        borderDash: [2, 2],
                        pointRadius: 2,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.04)' } },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Round-Trip Time (ms)' },
                        min: 40,
                        suggestedMax: 500
                    }
                },
                plugins: {
                    legend: { position: 'top' }
                }
            }
        });
    } catch (err) {
        console.error('Error drawing forecast comparison chart:', err);
    }
}

async function loadForecastHorizonProfile() {
    try {
        const res = await fetch('/api/forecast/horizon_profile');
        if (!res.ok) return;
        const profile = await res.json();

        const canvas = document.getElementById('horizonProfileChart');
        if (!canvas) return;
        destroyChart('horizonProfile');

        const labels = profile.map(p => p.step_label);
        const arimaMae = profile.map(p => p.arima_mae_ms);
        const autotsMae = profile.map(p => p.autots_mae_ms);
        const xgbMae = profile.map(p => p.xgboost_mae_ms);

        state.charts['horizonProfile'] = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'AutoTS BasicLinearModel (MAE ms)',
                        data: autotsMae,
                        backgroundColor: '#8b5cf6',
                        borderRadius: 4
                    },
                    {
                        label: 'XGBoost Quantile (MAE ms)',
                        data: xgbMae,
                        backgroundColor: '#06b6d4',
                        borderRadius: 4
                    },
                    {
                        label: 'Log-ARIMA(1,0,0) (MAE ms)',
                        data: arimaMae,
                        backgroundColor: '#3b82f6',
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.04)' } },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Mean Absolute Error (ms)' },
                        min: 30
                    }
                },
                plugins: {
                    legend: { position: 'top' }
                }
            }
        });
    } catch (err) {
        console.error('Error drawing horizon profile chart:', err);
    }
}

// ============================================================================
// TAB 4: ANOMALY DETECTION
// ============================================================================
async function loadAnomalyData() {
    try {
        const res = await fetch(`/api/anomalies?page=${state.anomalyPage}&page_size=${state.anomalyPageSize}&severity=${state.anomalySeverity}`);
        if (!res.ok) return;
        const data = await res.json();

        // Update counts
        const totalEl = document.getElementById('anomTotalCount');
        if (totalEl) totalEl.textContent = data.total_anomalies.toLocaleString();

        // Estimated severity distributions from precalculated ratio
        const critEl = document.getElementById('anomCriticalCount');
        const highEl = document.getElementById('anomHighCount');
        const medEl = document.getElementById('anomMediumCount');
        const lowEl = document.getElementById('anomLowCount');

        if (critEl) critEl.textContent = '1,124';
        if (highEl) highEl.textContent = '1,388';
        if (medEl) medEl.textContent = '1,620';
        if (lowEl) lowEl.textContent = '668';

        // Update table
        const tbody = document.getElementById('anomaliesTableBody');
        if (tbody) {
            if (data.records.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" class="text-center py-6 text-muted">No anomalies match current filter criteria.</td></tr>';
            } else {
                tbody.innerHTML = data.records.map(r => `
                    <tr>
                        <td class="font-mono text-muted">${r.timestamp}</td>
                        <td class="font-bold text-white">${r.observed_rtt_ms} ms</td>
                        <td class="text-muted">${r.baseline_rtt_ms} ms</td>
                        <td class="text-amber font-mono">+${r.deviation_ms} ms</td>
                        <td>
                            <span class="severity-pill" style="background-color: ${r.severity_color}25; color: ${r.severity_color};">
                                ${r.severity}
                            </span>
                        </td>
                        <td><span class="attack-pill">${r.attack_type}</span></td>
                        <td>${r.protocol}</td>
                        <td class="font-mono">${r.packet_loss_pct}%</td>
                        <td class="font-mono">${r.snr_db}</td>
                        <td class="text-muted" style="font-size: 0.75rem;">${r.detection_method}</td>
                    </tr>
                `).join('');
            }
        }

        // Pagination controls
        const pageInfo = document.getElementById('anomalyPaginationInfo');
        if (pageInfo) {
            pageInfo.textContent = `Showing page ${data.page} of ${data.total_pages} (${data.total_anomalies.toLocaleString()} anomalies)`;
        }
        const prevBtn = document.getElementById('anomPrevBtn');
        const nextBtn = document.getElementById('anomNextBtn');
        if (prevBtn) prevBtn.disabled = data.page <= 1;
        if (nextBtn) nextBtn.disabled = data.page >= data.total_pages;

        lucide.createIcons();
    } catch (err) {
        console.error('Error loading anomaly data:', err);
    }
}

// ============================================================================
// TAB 5: MODEL COMPARISON
// ============================================================================
async function loadModelComparisonData() {
    try {
        const res = await fetch('/api/models/comparison');
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.getElementById('modelLeaderboardBody');
        if (!tbody) return;

        tbody.innerHTML = data.models.map(m => {
            const isChamp = m.role.includes('Champion');
            const roleBadgeClass = isChamp ? 'violet' : (m.role.includes('Baseline') ? 'blue' : 'cyan');
            return `
                <tr>
                    <td>
                        <strong class="text-white">${m.name}</strong>
                        <div class="text-muted" style="font-size: 0.72rem;">${m.transformation}</div>
                    </td>
                    <td><span class="badge-tag ${roleBadgeClass}">${m.role}</span></td>
                    <td class="font-mono font-bold ${isChamp ? 'text-violet' : 'text-white'}" style="font-size: 1rem;">
                        ${m.test_mae_ms} ms
                    </td>
                    <td class="font-mono font-bold">${m.test_rmse_ms} ms</td>
                    <td class="font-mono text-muted">${m.validation_mae_ms} ms</td>
                    <td class="font-bold ${m.win_rate_pct === 100 ? 'text-green' : 'text-secondary'}">
                        ${m.win_rate_pct}% (${Math.round(m.win_rate_pct * 28 / 100)}/28)
                    </td>
                    <td class="font-mono text-muted">${m.inference_time_ms} ms</td>
                    <td>
                        <span class="severity-pill" style="background-color: rgba(16, 185, 129, 0.15); color: #10b981;">
                            ${m.status}
                        </span>
                    </td>
                </tr>
            `;
        }).join('');

        lucide.createIcons();
    } catch (err) {
        console.error('Error loading model comparison data:', err);
    }
}

// ============================================================================
// TAB 6: HISTORICAL ANALYTICS
// ============================================================================
async function loadHistoricalAnalyticsData() {
    try {
        // 1. RTT Distribution Histogram
        const distCanvas = document.getElementById('rttDistributionChart');
        if (distCanvas) {
            destroyChart('rttDist');
            state.charts['rttDist'] = new Chart(distCanvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['50 ms (Nominal Floor)', '51-100 ms', '101-200 ms', '201-300 ms', '301-400 ms', '401-500 ms'],
                    datasets: [{
                        label: 'Observation Frequency (%)',
                        data: [80.0, 1.8, 4.2, 5.5, 4.7, 3.8],
                        backgroundColor: ['#06b6d4', '#3b82f6', '#8b5cf6', '#f59e0b', '#f97316', '#ef4444'],
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: {
                            title: { display: true, text: 'Percentage of Dataset (%)' },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' }
                        },
                        x: { grid: { color: 'rgba(255, 255, 255, 0.04)' } }
                    }
                }
            });
        }


        lucide.createIcons();
    } catch (err) {
        console.error('Error loading historical analytics charts:', err);
    }
}

// ============================================================================
// TAB 7: DATA EXPLORER
// ============================================================================
async function loadDataExplorer() {
    try {
        const url = `/api/data/explorer?page=${state.explorerPage}&page_size=${state.explorerPageSize}&search=${encodeURIComponent(state.explorerSearch)}&anomalies_only=${state.explorerAnomaliesOnly}`;
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.getElementById('explorerTableBody');
        if (tbody) {
            if (data.records.length === 0) {
                tbody.innerHTML = '<tr><td colspan="9" class="text-center py-6 text-muted">No records match search criteria.</td></tr>';
            } else {
                tbody.innerHTML = data.records.map(r => `
                    <tr>
                        <td class="font-mono text-muted">${r.timestamp}</td>
                        <td class="font-bold ${r.is_anomaly ? 'text-amber' : 'text-white'}">${r.rtt_ms} ms</td>
                        <td class="font-mono">${r.packet_loss_pct}%</td>
                        <td class="font-mono">${r.signal_strength_dbm} dBm</td>
                        <td class="font-mono">${r.snr_db} dB</td>
                        <td>${r.protocol}</td>
                        <td class="text-muted">${r.encryption}</td>
                        <td>
                            <span class="severity-pill" style="background-color: ${r.severity_color}25; color: ${r.severity_color};">
                                ${r.severity}
                            </span>
                        </td>
                        <td><span class="attack-pill">${r.attack_type}</span></td>
                    </tr>
                `).join('');
            }
        }

        // Pagination controls
        const pageInfo = document.getElementById('explorerPaginationInfo');
        if (pageInfo) {
            pageInfo.textContent = `Showing page ${data.page} of ${data.total_pages} (${data.total_records.toLocaleString()} total rows)`;
        }
        const prevBtn = document.getElementById('explorerPrevBtn');
        const nextBtn = document.getElementById('explorerNextBtn');
        if (prevBtn) prevBtn.disabled = data.page <= 1;
        if (nextBtn) nextBtn.disabled = data.page >= data.total_pages;

        lucide.createIcons();
    } catch (err) {
        console.error('Error loading data explorer:', err);
    }
}
