/* ═══════════════════════════════════════════════════════
   ChurnIQ — app.js
   ═══════════════════════════════════════════════════════ */

// ── Tab Navigation ────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const id = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + id).classList.add('active');
    if (id === 'insights' && !featureChartReady) initFeatureChart();
  });
});

// ── Feature Weights (Dynamically Loaded) ──────────────────
// Fallback values match the normalised coefficients from generate_data.py.
// They are identical to the contents of model/feature_weights.json so the
// app behaves the same whether or not the fetch succeeds.
let featureWeights = {
  lastLogin: 0.190,
  nps: 0.143,
  tickets: 0.119,
  tenure: 0.095,
  onboarding: 0.095,
  contract: 0.071,
  modules: 0.071,
  industry: 0.071,
  spend: 0.048,
  csm: 0.048,
  users: 0.048
};

fetch('model/feature_weights.json')
  .then(res => res.json())
  .then(data => {
    featureWeights = data;
    updatePrediction();
  })
  .catch(err => console.warn('Using default weights. Could not load model/feature_weights.json:', err));

// ── Chart global defaults ─────────────────────────────────
Chart.defaults.color          = '#7a7167';
Chart.defaults.borderColor    = '#272320';
Chart.defaults.font.family    = "'Inter', sans-serif";
Chart.defaults.font.size      = 11;
Chart.defaults.plugins.legend.labels.boxWidth = 10;
Chart.defaults.plugins.legend.labels.padding  = 14;

const ACCENT  = '#c9f531';
const DANGER  = '#ff5252';
const WARN    = '#ff9f2e';
const SUCCESS = '#3dffa0';
const MUTED   = '#7a7167';
const S3      = '#201e1b';
const GRID    = '#272320';

const chartOpts = {
  responsive: true,
  animation: { duration: 700, easing: 'easeOutQuart' },
  plugins: { legend: { display: false }, tooltip: { enabled: true } },
};

// ── Dashboard Charts ──────────────────────────────────────

// Contract type
new Chart(document.getElementById('chartContract'), {
  type: 'bar',
  data: {
    labels: ['Monthly', 'Annual', 'Multi-year'],
    datasets: [{
      data: [44.2, 21.8, 9.5],
      backgroundColor: [DANGER + 'cc', WARN + 'cc', ACCENT + 'cc'],
      borderColor:     [DANGER,         WARN,         ACCENT],
      borderWidth: 1,
      borderRadius: 2,
      borderSkipped: false,
    }]
  },
  options: {
    ...chartOpts,
    scales: {
      y: { beginAtZero: true, max: 55, ticks: { callback: v => v + '%' }, grid: { color: GRID } },
      x: { grid: { display: false } },
    },
  },
});

// Industry
new Chart(document.getElementById('chartIndustry'), {
  type: 'bar',
  data: {
    labels: ['Logistics', 'HR-Tech', 'Fintech', 'Healthcare', 'Retail'],
    datasets: [{
      data: [24.8, 28.3, 22.1, 31.5, 33.2],
      backgroundColor: ['#c9f531bb','#3dffa0bb','#ff9f2ebb','#ff5252bb','#60a5fabb'],
      borderColor:     ['#c9f531',  '#3dffa0',  '#ff9f2e',  '#ff5252',  '#60a5fa'],
      borderWidth: 1,
      borderRadius: 2,
      borderSkipped: false,
    }]
  },
  options: {
    ...chartOpts,
    scales: {
      y: { beginAtZero: true, max: 45, ticks: { callback: v => v + '%' }, grid: { color: GRID } },
      x: { grid: { display: false } },
    },
  },
});

// Tenure
new Chart(document.getElementById('chartTenure'), {
  type: 'bar',
  data: {
    labels: ['0–6 mo', '7–12 mo', '13–24 mo', '25–36 mo', '37–48 mo', '49–60 mo'],
    datasets: [
      {
        label: 'Clients',
        data: [120, 180, 280, 210, 140, 70],
        backgroundColor: ACCENT + '22',
        borderColor: ACCENT + '88',
        borderWidth: 1,
        borderRadius: 2,
        borderSkipped: false,
        yAxisID: 'yCount',
      },
      {
        label: 'Churn %',
        data: [52.1, 38.4, 24.7, 15.2, 10.3, 7.8],
        type: 'line',
        borderColor: DANGER,
        backgroundColor: DANGER + '18',
        fill: true,
        tension: 0.45,
        pointRadius: 3,
        pointBackgroundColor: DANGER,
        pointBorderColor: '#090807',
        pointBorderWidth: 1.5,
        yAxisID: 'yRate',
      },
    ],
  },
  options: {
    ...chartOpts,
    plugins: { legend: { display: true, position: 'top',
      labels: { color: MUTED, boxWidth: 10, padding: 14 } } },
    scales: {
      yCount: {
        position: 'left',
        beginAtZero: true,
        grid: { color: GRID },
        title: { display: true, text: 'Clients', color: MUTED, font: { size: 10 } },
      },
      yRate: {
        position: 'right',
        beginAtZero: true,
        max: 70,
        grid: { display: false },
        ticks: { callback: v => v + '%' },
        title: { display: true, text: 'Churn %', color: MUTED, font: { size: 10 } },
      },
      x: { grid: { display: false } },
    },
  },
});

// NPS vs Churn
new Chart(document.getElementById('chartNPS'), {
  type: 'line',
  data: {
    labels: ['0','1','2','3','4','5','6','7','8','9','10'],
    datasets: [{
      data: [72.4, 68.1, 61.3, 53.8, 44.2, 34.6, 24.1, 16.8, 10.5, 6.2, 3.9],
      borderColor: ACCENT,
      backgroundColor: ACCENT + '14',
      fill: true,
      tension: 0.45,
      pointRadius: 3,
      pointBackgroundColor: ACCENT,
      pointBorderColor: '#090807',
      pointBorderWidth: 1.5,
    }],
  },
  options: {
    ...chartOpts,
    scales: {
      y: {
        beginAtZero: true,
        max: 85,
        ticks: { callback: v => v + '%' },
        grid: { color: GRID },
        title: { display: true, text: 'Churn Rate', color: MUTED, font: { size: 10 } },
      },
      x: {
        grid: { display: false },
        title: { display: true, text: 'NPS Score', color: MUTED, font: { size: 10 } },
      },
    },
  },
});

// ── Feature Importance Chart ──────────────────────────────
let featureChartReady = false;
function initFeatureChart() {
  featureChartReady = true;
  const labelsMap = {
    lastLogin: 'Last Login Days',
    nps: 'NPS Score',
    tickets: 'Support Tickets',
    spend: 'Monthly Spend',
    tenure: 'Tenure',
    modules: 'Product Modules',
    users: 'Num. of Users',
    contract: 'Contract Type',
    onboarding: 'Onboarding Done',
    csm: 'Has CSM',
    industry: 'Industry'
  };

  const data = Object.keys(featureWeights).map(k => ({
    label: labelsMap[k] || k,
    val: featureWeights[k]
  })).sort((a, b) => b.val - a.val);

  new Chart(document.getElementById('chartFeatures'), {
    type: 'bar',
    data: {
      labels: data.map(d => d.label),
      datasets: [{
        data: data.map(d => d.val),
        backgroundColor: data.map((_, i) => lerpColor('#ff5252', '#c9f531', i / (data.length - 1)) + 'cc'),
        borderColor:     data.map((_, i) => lerpColor('#ff5252', '#c9f531', i / (data.length - 1))),
        borderWidth: 1,
        borderRadius: 2,
        borderSkipped: false,
      }],
    },
    options: {
      ...chartOpts,
      indexAxis: 'y',
      scales: {
        x: {
          beginAtZero: true,
          ticks: { callback: v => (v * 100).toFixed(0) + '%' },
          grid: { color: GRID },
        },
        y: { grid: { display: false } },
      },
    },
  });
}

function lerpColor(a, b, t) {
  const ah = parseInt(a.replace('#',''), 16);
  const bh = parseInt(b.replace('#',''), 16);
  const ar = (ah>>16)&0xff, ag = (ah>>8)&0xff, ab = ah&0xff;
  const br = (bh>>16)&0xff, bg = (bh>>8)&0xff, bb = bh&0xff;
  return `#${[
    Math.round(ar+(br-ar)*t),
    Math.round(ag+(bg-ag)*t),
    Math.round(ab+(bb-ab)*t),
  ].map(v => v.toString(16).padStart(2,'0')).join('')}`;
}

// ── Live Prediction ───────────────────────────────────────
function getInputs() {
  return {
    tenure:    +document.getElementById('tenure').value,
    spend:     +document.getElementById('spend').value,
    users:     +document.getElementById('users').value,
    tickets:   +document.getElementById('tickets').value,
    modules:   +document.getElementById('modules').value,
    lastLogin: +document.getElementById('lastlogin').value,
    nps:       +document.getElementById('nps').value,
    contract:  document.getElementById('contract').value,
    industry:  document.getElementById('industry').value,
    onboarding: document.querySelector('input[name="onboarding"]:checked').value === 'yes',
    csm:        document.querySelector('input[name="csm"]:checked').value === 'yes',
  };
}

function updateSliderDisplays(f) {
  document.getElementById('v-tenure').innerHTML   = `${f.tenure} <small>mo</small>`;
  document.getElementById('v-spend').textContent   = `€${f.spend.toLocaleString()}`;
  document.getElementById('v-users').textContent   = f.users;
  document.getElementById('v-tickets').textContent = f.tickets;
  document.getElementById('v-modules').textContent = f.modules;
  document.getElementById('v-lastlogin').innerHTML = `${f.lastLogin} <small>days ago</small>`;
  document.getElementById('v-nps').innerHTML       = `${f.nps} <small>/ 10</small>`;
}

function scoreChurn(f) {
  // Normalize features so 1 = highest churn risk, 0 = lowest churn risk
  const nLastLogin = f.lastLogin / 120;
  const nNPS       = (10 - f.nps) / 10;
  const nTickets   = f.tickets / 30;
  const nSpend     = (10000 - f.spend) / 9500;
  const nTenure    = (60 - f.tenure) / 59;
  const nModules   = (10 - f.modules) / 9;
  const nUsers     = (500 - f.users) / 499;
  
  let nContract = 0.5; // Annual
  if (f.contract === 'Monthly') nContract = 1.0;
  else if (f.contract === 'Multi-year') nContract = 0.0;
  
  const nOnboarding = f.onboarding ? 0.0 : 1.0;
  const nCSM        = f.csm ? 0.0 : 1.0;
  
  let nIndustry = 0.5;
  if (f.industry === 'Retail') nIndustry = 1.0;
  else if (f.industry === 'Healthcare') nIndustry = 0.8;
  else if (f.industry === 'Logistics') nIndustry = 0.6;
  else if (f.industry === 'Fintech') nIndustry = 0.3;
  else if (f.industry === 'HR-Tech') nIndustry = 0.0;

  // Weighted sum using dynamically loaded feature importance
  const z_raw = 
    (nLastLogin * featureWeights.lastLogin) +
    (nNPS       * featureWeights.nps) +
    (nTickets   * featureWeights.tickets) +
    (nSpend     * featureWeights.spend) +
    (nTenure    * featureWeights.tenure) +
    (nModules   * featureWeights.modules) +
    (nUsers     * featureWeights.users) +
    (nContract  * featureWeights.contract) +
    (nOnboarding* featureWeights.onboarding) +
    (nCSM       * featureWeights.csm) +
    (nIndustry  * featureWeights.industry);

  // Map the raw linear sum to a probability using a sigmoid S-curve.
  // We set the inflection point at 0.45 so average clients get ~10-25% risk.
  const score = Math.round(100 / (1 + Math.exp(-12 * (z_raw - 0.45))));
  
  // Bound it just in case between 1 and 99
  return Math.max(1, Math.min(99, score));
}

function getRiskFactors(f) {
  const rows = [];
  rows.push({ label: 'Last login', val: f.lastLogin + ' days ago', bad: f.lastLogin > 60 });
  rows.push({ label: 'NPS Score',  val: f.nps + ' / 10',           bad: f.nps < 5 });
  rows.push({ label: 'Support tickets (90d)', val: f.tickets,      bad: f.tickets > 10 });
  rows.push({ label: 'Onboarding', val: f.onboarding ? 'Done' : 'Incomplete', bad: !f.onboarding });
  rows.push({ label: 'Contract',   val: f.contract,                 bad: f.contract === 'Monthly' });
  rows.push({ label: 'CSM assigned', val: f.csm ? 'Yes' : 'No',   bad: !f.csm });
  return rows;
}

function getActions(pct) {
  if (pct >= 70) return [
    { icon: '🚨', text: 'Schedule an executive business review within 48 hours' },
    { icon: '🔧', text: 'Assign senior CSM to audit product adoption blockers' },
    { icon: '💡', text: 'Offer a custom success plan or meaningful incentive' },
  ];
  if (pct >= 40) return [
    { icon: '📧', text: 'Send a personalised check-in from the account team' },
    { icon: '📊', text: 'Share a health dashboard highlighting unused features' },
    { icon: '🎯', text: 'Propose a QBR to re-align on business outcomes' },
  ];
  return [
    { icon: '🤝', text: 'Explore upsell or expansion opportunities' },
    { icon: '⭐', text: 'Invite to reference customer or case-study programme' },
    { icon: '🔄', text: 'Schedule a regular touchpoint to maintain momentum' },
  ];
}

function updatePrediction() {
  const f = getInputs();
  updateSliderDisplays(f);
  const pct = scoreChurn(f);

  const riskClass = pct >= 70 ? 'high' : pct >= 40 ? 'med' : 'low';
  const riskText  = pct >= 70 ? '⬤ High Risk' : pct >= 40 ? '◆ Medium Risk' : '● Low Risk';
  const color     = pct >= 70 ? DANGER : pct >= 40 ? WARN : SUCCESS;

  // Label
  const lbl = document.getElementById('risk-level-label');
  lbl.textContent = riskText;
  lbl.className   = 'risk-level-label ' + riskClass;

  // Percentage
  const pctEl = document.getElementById('risk-pct');
  pctEl.textContent = pct + '%';
  pctEl.className   = 'risk-pct ' + riskClass;

  // Thermometer
  const fill = document.getElementById('thermo-fill');
  fill.style.height     = Math.min(pct, 100) + '%';
  fill.style.background = color;
  fill.style.boxShadow  = `0 0 12px ${color}55`;

  // Risk factors
  const factors = getRiskFactors(f);
  document.getElementById('risk-factors').innerHTML = factors.map(r => `
    <div class="risk-factor-row">
      <span class="rf-label">${r.label}</span>
      <span class="rf-badge ${r.bad ? 'bad' : 'ok'}">${r.val}</span>
    </div>
  `).join('');

  // Actions
  const actions = getActions(pct);
  document.getElementById('actions-section').innerHTML = `
    <div class="actions-title">Recommended Actions</div>
    ${actions.map(a => `
      <div class="action-item">
        <span class="action-icon">${a.icon}</span>
        <span>${a.text}</span>
      </div>
    `).join('')}
  `;
}

// ── Wire up all inputs ────────────────────────────────────
['tenure','spend','users','tickets','modules','lastlogin','nps'].forEach(id => {
  document.getElementById(id).addEventListener('input', updatePrediction);
});
document.getElementById('contract').addEventListener('change', updatePrediction);
document.getElementById('industry').addEventListener('change', updatePrediction);
document.querySelectorAll('input[name="onboarding"], input[name="csm"]').forEach(el => {
  el.addEventListener('change', updatePrediction);
});

// Run once on load
updatePrediction();
