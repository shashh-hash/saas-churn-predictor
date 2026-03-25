/* ═══════════════════════════════════════════════════════════
   B2B SaaS Churn Predictor – app.js
   ═══════════════════════════════════════════════════════════ */

// ── Tab Navigation ─────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
    // Lazy-init charts when switching to their tab
    if (tab === 'insights' && !featureChartInit) initFeatureChart();
  });
});

// ── Slider Sync ────────────────────────────────────────────
function syncVal(input, spanId) {
  document.getElementById(spanId).textContent = input.value;
}

// ── Chart Defaults ─────────────────────────────────────────
Chart.defaults.color = '#8892aa';
Chart.defaults.borderColor = '#2e3250';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

const ACCENT   = '#7c6cf8';
const ACCENT2  = '#56cfb2';
const DANGER   = '#f76b6b';
const WARN     = '#f5a623';
const SURFACE2 = '#22263a';

// ── Dashboard Charts ───────────────────────────────────────

// Churn Rate by Contract Type
new Chart(document.getElementById('chartContract'), {
  type: 'bar',
  data: {
    labels: ['Monthly', 'Annual', 'Multi-year'],
    datasets: [{
      label: 'Churn Rate (%)',
      data: [44.2, 21.8, 9.5],
      backgroundColor: [DANGER, WARN, ACCENT2],
      borderRadius: 6,
      borderSkipped: false,
    }]
  },
  options: {
    plugins: { legend: { display: false } },
    scales: {
      y: {
        beginAtZero: true,
        max: 60,
        ticks: { callback: v => v + '%' },
        grid: { color: '#2e3250' },
      },
      x: { grid: { display: false } }
    },
    animation: { duration: 900, easing: 'easeOutQuart' }
  }
});

// Churn Rate by Industry
new Chart(document.getElementById('chartIndustry'), {
  type: 'bar',
  data: {
    labels: ['Logistics', 'HR-Tech', 'Fintech', 'Healthcare', 'Retail'],
    datasets: [{
      label: 'Churn Rate (%)',
      data: [24.8, 28.3, 22.1, 31.5, 33.2],
      backgroundColor: ['#7c6cf8','#56cfb2','#f5a623','#f76b6b','#60a5fa'],
      borderRadius: 6,
      borderSkipped: false,
    }]
  },
  options: {
    plugins: { legend: { display: false } },
    scales: {
      y: {
        beginAtZero: true,
        max: 45,
        ticks: { callback: v => v + '%' },
        grid: { color: '#2e3250' },
      },
      x: { grid: { display: false } }
    },
    animation: { duration: 900, easing: 'easeOutQuart' }
  }
});

// Tenure Buckets vs Churn Rate
new Chart(document.getElementById('chartTenure'), {
  type: 'bar',
  data: {
    labels: ['0–6 mo', '7–12 mo', '13–24 mo', '25–36 mo', '37–48 mo', '49–60 mo'],
    datasets: [
      {
        label: 'Client Count',
        data: [120, 180, 280, 210, 140, 70],
        backgroundColor: ACCENT + '55',
        borderColor: ACCENT,
        borderWidth: 1,
        borderRadius: 4,
        yAxisID: 'yCount',
      },
      {
        label: 'Churn Rate (%)',
        data: [52.1, 38.4, 24.7, 15.2, 10.3, 7.8],
        type: 'line',
        borderColor: DANGER,
        backgroundColor: DANGER + '22',
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: DANGER,
        yAxisID: 'yRate',
      }
    ]
  },
  options: {
    plugins: {
      legend: { position: 'top', labels: { boxWidth: 12 } }
    },
    scales: {
      yCount: {
        position: 'left',
        beginAtZero: true,
        title: { display: true, text: 'Clients', font: { size: 11 } },
        grid: { color: '#2e3250' },
      },
      yRate: {
        position: 'right',
        beginAtZero: true,
        max: 70,
        ticks: { callback: v => v + '%' },
        title: { display: true, text: 'Churn %', font: { size: 11 } },
        grid: { display: false },
      },
      x: { grid: { display: false } }
    },
    animation: { duration: 900 }
  }
});

// NPS Score vs Churn Rate
new Chart(document.getElementById('chartNPS'), {
  type: 'line',
  data: {
    labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    datasets: [{
      label: 'Churn Rate (%)',
      data: [72.4, 68.1, 61.3, 53.8, 44.2, 34.6, 24.1, 16.8, 10.5, 6.2, 3.9],
      borderColor: ACCENT2,
      backgroundColor: ACCENT2 + '22',
      fill: true,
      tension: 0.45,
      pointRadius: 4,
      pointBackgroundColor: ACCENT2,
    }]
  },
  options: {
    plugins: { legend: { display: false } },
    scales: {
      y: {
        beginAtZero: true,
        max: 90,
        ticks: { callback: v => v + '%' },
        grid: { color: '#2e3250' },
        title: { display: true, text: 'Churn Rate', font: { size: 11 } },
      },
      x: {
        grid: { display: false },
        title: { display: true, text: 'NPS Score', font: { size: 11 } },
      }
    },
    animation: { duration: 900 }
  }
});

// ── Feature Importance Chart ────────────────────────────────
let featureChartInit = false;

function initFeatureChart() {
  featureChartInit = true;
  const features = [
    'Last Login Days',
    'NPS Score',
    'Support Tickets',
    'Monthly Spend',
    'Tenure',
    'Product Modules',
    'Num. of Users',
    'Contract Type',
    'Onboarding Done',
    'Has CSM',
    'Industry',
  ];
  const importances = [0.278, 0.192, 0.148, 0.091, 0.083, 0.071, 0.052, 0.038, 0.027, 0.011, 0.009];

  // Sort descending
  const sorted = features.map((f, i) => ({ f, v: importances[i] }))
    .sort((a, b) => b.v - a.v);

  new Chart(document.getElementById('chartFeatures'), {
    type: 'bar',
    data: {
      labels: sorted.map(d => d.f),
      datasets: [{
        label: 'Importance',
        data: sorted.map(d => d.v),
        backgroundColor: sorted.map((_, i) => {
          const t = i / (sorted.length - 1);
          return lerpColor('#7c6cf8', '#56cfb2', t);
        }),
        borderRadius: 5,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { callback: v => (v * 100).toFixed(0) + '%' },
          grid: { color: '#2e3250' },
        },
        y: { grid: { display: false } }
      },
      animation: { duration: 1000, easing: 'easeOutQuart' }
    }
  });
}

function lerpColor(a, b, t) {
  const ah = parseInt(a.replace('#',''), 16);
  const bh = parseInt(b.replace('#',''), 16);
  const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
  const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
  const rr = Math.round(ar + (br - ar) * t);
  const rg = Math.round(ag + (bg - ag) * t);
  const rb = Math.round(ab + (bb - ab) * t);
  return `rgb(${rr},${rg},${rb})`;
}

// ── Prediction Logic ───────────────────────────────────────
/*
  Mirrors the churn scoring in generate_data.py:
    tenure < 6      → +2
    tickets > 10    → +2
    lastLogin > 60  → +3   (strongest signal)
    nps < 5         → +2
    contract=Monthly→ +1
    onboarding=No   → +2
    modules < 3     → +1
    csm=No          → +1
  max = 14
*/
function predictChurn(f) {
  let score = 0;
  if (f.tenure < 6)              score += 2;
  if (f.tickets > 10)            score += 2;
  if (f.lastLogin > 60)          score += 3;
  if (f.nps < 5)                 score += 2;
  if (f.contract === 'Monthly')  score += 1;
  if (!f.onboarding)             score += 2;
  if (f.modules < 3)             score += 1;
  if (!f.csm)                    score += 1;

  // Smooth probability with a slight sigmoid-like curve so mid-range
  // scores don't feel too binary
  const raw = score / 14;
  return Math.round(sigmoidScale(raw) * 100);
}

function sigmoidScale(x) {
  // Maps [0,1] → [0,1] with gentle S-curve
  return 1 / (1 + Math.exp(-10 * (x - 0.45)));
}

let gaugeChart = null;

function runPrediction() {
  const f = {
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

  const pct = predictChurn(f);

  // Risk level
  let riskClass, riskText, actions;
  if (pct >= 70) {
    riskClass = 'risk-high';
    riskText  = '🔴 HIGH RISK';
    actions   = [
      '📞 Schedule an executive business review within 48 hours',
      '🔧 Assign a senior CSM to audit product adoption blockers',
      '💡 Offer a custom success plan or usage incentive',
    ];
  } else if (pct >= 40) {
    riskClass = 'risk-medium';
    riskText  = '🟠 MEDIUM RISK';
    actions   = [
      '📧 Send a personalised check-in email from the account team',
      '📊 Share a health dashboard highlighting unused features',
      '🎯 Propose a QBR to re-align on business outcomes',
    ];
  } else {
    riskClass = 'risk-low';
    riskText  = '🟢 LOW RISK';
    actions   = [
      '🤝 Explore upsell or expansion opportunities',
      '⭐ Invite to reference customer or case-study programme',
      '🔄 Schedule a regular touchpoint to maintain engagement',
    ];
  }

  // Show result panel
  document.querySelector('.result-placeholder').classList.add('hidden');
  const rc = document.getElementById('result-content');
  rc.classList.remove('hidden');

  document.getElementById('result-label').textContent  = riskText;
  document.getElementById('result-label').className    = 'result-risk-label ' + riskClass;
  document.getElementById('result-percent').textContent = pct + '%';

  const color = pct >= 70 ? DANGER : pct >= 40 ? WARN : ACCENT2;
  document.getElementById('result-percent').style.color = color;

  // Actions
  const box = document.getElementById('actions-box');
  box.innerHTML = `<h5>Recommended Actions</h5><ul>${actions.map(a => `<li>✦ <span>${a}</span></li>`).join('')}</ul>`;

  // Gauge
  drawGauge(pct, color);
}

function drawGauge(pct, color) {
  const ctx = document.getElementById('gaugeChart').getContext('2d');

  if (gaugeChart) { gaugeChart.destroy(); }

  gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [pct, 100 - pct],
        backgroundColor: [color, SURFACE2],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
      }]
    },
    options: {
      cutout: '70%',
      responsive: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      },
      animation: { animateRotate: true, duration: 700 }
    },
    plugins: [{
      id: 'gaugeLabel',
      afterDraw(chart) {
        const { ctx, chartArea: { left, right, top, bottom } } = chart;
        const cx = (left + right) / 2;
        const cy = bottom - 8;
        ctx.save();
        ctx.font = 'bold 22px Segoe UI, sans-serif';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(pct + '%', cx, cy);
        ctx.restore();
      }
    }]
  });
}
