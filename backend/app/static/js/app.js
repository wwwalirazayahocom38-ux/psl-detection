/**
 * Pakistani Sign Language (PSL) Real-Time AI Detection System
 * Frontend Application Core
 */

// Application State
const state = {
  isCameraRunning: false,
  isAudioEnabled: true,
  webSocket: null,
  hands: null,
  camera: null,
  activeTab: 'detect',
  allSigns: [],
  recentDetections: [],
  sentenceTokens: [],
  lastSpokenSign: null,
  lastSpokenTimestamp: 0,
  latencyTimes: [],
};

// DOM Elements Cache
const elements = {
  video: document.getElementById('webcam-video'),
  canvas: document.getElementById('output-canvas'),
  canvasCtx: null,
  btnToggleCamera: document.getElementById('btn-toggle-camera'),
  cameraBtnText: document.getElementById('camera-btn-text'),
  btnQuickSample: document.getElementById('btn-quick-sample'),
  connectionStatus: document.getElementById('connection-status'),
  ttsToggle: document.getElementById('tts-toggle'),
  btnSpeakSign: document.getElementById('btn-speak-sign'),
  btnClearHistory: document.getElementById('btn-clear-history'),
  btnCopySentence: document.getElementById('btn-copy-sentence'),
  handTrackingTag: document.getElementById('hand-tracking-tag'),
  trackingStatusText: document.getElementById('tracking-status-text'),
  detectedUrdu: document.getElementById('detected-urdu'),
  detectedEnglish: document.getElementById('detected-english'),
  detectedPhonetic: document.getElementById('detected-phonetic'),
  signCategoryBadge: document.getElementById('sign-category-badge'),
  confidencePercentage: document.getElementById('confidence-percentage'),
  confidenceBar: document.getElementById('confidence-bar'),
  sentenceContent: document.getElementById('sentence-content'),
  historyList: document.getElementById('history-list'),
  historyCount: document.getElementById('history-count'),
  metricLandmarks: document.getElementById('metric-landmarks'),
  metricBuffer: document.getElementById('metric-buffer'),
  metricLatency: document.getElementById('metric-latency'),
  toast: document.getElementById('toast'),
  signsGrid: document.getElementById('signs-grid'),
};

if (elements.canvas) {
  elements.canvasCtx = elements.canvas.getContext('2d');
}

/**
 * Initialize Application
 */
async function initApp() {
  setupNavigation();
  setupAudioControls();
  setupWebSocket();
  loadAllSigns();
  setupDictionaryFilters();
  setupQuickTest();
  setupMediaPipe();

  // Load initial analytics if on analytics tab
  loadAnalytics();

  showToast("PSL Detection System initialized.");
}

/**
 * Toast Notification Utility
 */
function showToast(message) {
  if (!elements.toast) return;
  elements.toast.textContent = message;
  elements.toast.classList.add('show');
  setTimeout(() => {
    elements.toast.classList.remove('show');
  }, 2800);
}

/**
 * Tab Navigation
 */
function setupNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.getAttribute('data-tab');
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      document.querySelectorAll('.tab-view').forEach(view => {
        view.classList.remove('active');
      });

      const activeView = document.getElementById(`view-${target}`);
      if (activeView) activeView.classList.add('active');
      state.activeTab = target;

      if (target === 'analytics') {
        loadAnalytics();
      }
    });
  });

  const refreshBtn = document.getElementById('btn-refresh-analytics');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', loadAnalytics);
  }
}

/**
 * Text-to-Speech (Urdu & English)
 */
function setupAudioControls() {
  if (elements.ttsToggle) {
    elements.ttsToggle.addEventListener('click', () => {
      state.isAudioEnabled = !state.isAudioEnabled;
      elements.ttsToggle.classList.toggle('active', state.isAudioEnabled);
      const textSpan = elements.ttsToggle.querySelector('.tts-text');
      const iconSpan = elements.ttsToggle.querySelector('.tts-icon');
      if (state.isAudioEnabled) {
        textSpan.textContent = 'Audio On';
        iconSpan.textContent = '🔊';
        showToast('Text-to-Speech enabled (Urdu pronunciation)');
      } else {
        textSpan.textContent = 'Muted';
        iconSpan.textContent = '🔇';
        showToast('Text-to-Speech muted');
      }
    });
  }

  if (elements.btnSpeakSign) {
    elements.btnSpeakSign.addEventListener('click', () => {
      const urduText = elements.detectedUrdu.textContent.trim();
      const phonetic = elements.detectedPhonetic.textContent.replace('Phonetic: /', '').replace('/', '').trim();
      speakText(urduText, phonetic, true);
    });
  }

  if (elements.btnClearHistory) {
    elements.btnClearHistory.addEventListener('click', () => {
      state.recentDetections = [];
      state.sentenceTokens = [];
      renderHistory();
      renderSentence();
      showToast('Detection history cleared');
    });
  }

  if (elements.btnCopySentence) {
    elements.btnCopySentence.addEventListener('click', () => {
      const text = state.sentenceTokens.map(t => t.urdu).join(' ');
      if (!text) {
        showToast('No translation sequence to copy');
        return;
      }
      navigator.clipboard.writeText(text);
      showToast('Urdu translation copied to clipboard! 📋');
    });
  }
}

function speakText(urduText, phoneticText = '', force = false) {
  if (!state.isAudioEnabled && !force) return;
  if (!('speechSynthesis' in window)) return;

  const now = Date.now();
  if (!force && urduText === state.lastSpokenSign && now - state.lastSpokenTimestamp < 3500) {
    return; // suppress duplicate spam
  }

  state.lastSpokenSign = urduText;
  state.lastSpokenTimestamp = now;

  window.speechSynthesis.cancel(); // Stop ongoing speech

  // Prefer Urdu utterance
  const utterance = new SpeechSynthesisUtterance(urduText || phoneticText);
  utterance.rate = 0.88;
  utterance.pitch = 1.0;

  // Try selecting an Urdu or Hindi/Arabic voice if available
  const voices = window.speechSynthesis.getVoices();
  const urduVoice = voices.find(v => v.lang.startsWith('ur') || v.lang.startsWith('ar') || v.lang.startsWith('hi'));
  if (urduVoice) {
    utterance.voice = urduVoice;
    utterance.lang = urduVoice.lang;
  } else {
    // If no native Urdu voice, pronounce phonetic cues in standard voice
    if (phoneticText) {
      utterance.text = phoneticText;
    }
    utterance.lang = 'en-US';
  }

  window.speechSynthesis.speak(utterance);
}

/**
 * WebSocket Connection for Real-Time Streaming
 */
function setupWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/detect`;

  try {
    state.webSocket = new WebSocket(wsUrl);

    state.webSocket.onopen = () => {
      updateStatus(true, 'Connected (Real-Time WS)');
    };

    state.webSocket.onclose = () => {
      updateStatus(false, 'Disconnected (Reconnecting...)');
      setTimeout(setupWebSocket, 3000);
    };

    state.webSocket.onerror = () => {
      updateStatus(false, 'Connection Error');
    };

    state.webSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status === 'success') {
          handlePredictionResult(data);
        }
      } catch (e) {
        console.error('Error parsing WS message:', e);
      }
    };
  } catch (err) {
    console.warn('WebSocket init exception:', err);
    updateStatus(false, 'Offline Mode');
  }
}

function updateStatus(connected, label) {
  if (!elements.connectionStatus) return;
  elements.connectionStatus.classList.toggle('connected', connected);
  const textSpan = elements.connectionStatus.querySelector('.status-label');
  if (textSpan) textSpan.textContent = label;
}

/**
 * MediaPipe Hands Setup
 */
function setupMediaPipe() {
  if (typeof Hands === 'undefined') {
    console.warn('MediaPipe library not loaded yet, using simulated detection.');
    return;
  }

  try {
    state.hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    state.hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.65,
      minTrackingConfidence: 0.5
    });

    state.hands.onResults(onHandResults);

    if (elements.btnToggleCamera) {
      elements.btnToggleCamera.addEventListener('click', toggleCamera);
    }
  } catch (err) {
    console.error('MediaPipe initialization failed:', err);
  }
}

async function toggleCamera() {
  if (state.isCameraRunning) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    if (typeof Camera === 'undefined') {
      showToast('Camera utility not ready.');
      return;
    }

    state.camera = new Camera(elements.video, {
      onFrame: async () => {
        if (state.hands && state.isCameraRunning) {
          const t0 = performance.now();
          await state.hands.send({ image: elements.video });
          const t1 = performance.now();
          updateLatency(Math.round(t1 - t0));
        }
      },
      width: 640,
      height: 480
    });

    await state.camera.start();
    state.isCameraRunning = true;
    elements.cameraBtnText.textContent = 'Stop Camera';
    showToast('Webcam active. MediaPipe tracking hands...');
  } catch (err) {
    console.error('Camera access error:', err);
    showToast('Unable to open camera. Try "Quick Test Sign" button!');
    const fallback = document.getElementById('camera-fallback');
    if (fallback) fallback.style.display = 'flex';
  }
}

function stopCamera() {
  if (state.camera) {
    state.camera.stop();
  }
  state.isCameraRunning = false;
  elements.cameraBtnText.textContent = 'Start Camera';
  showToast('Webcam feed paused.');
}

function updateLatency(ms) {
  state.latencyTimes.push(ms);
  if (state.latencyTimes.length > 10) state.latencyTimes.shift();
  const avg = Math.round(state.latencyTimes.reduce((a, b) => a + b, 0) / state.latencyTimes.length);
  if (elements.metricLatency) {
    elements.metricLatency.textContent = `~${avg} ms`;
  }
}

/**
 * Handle MediaPipe Hand Landmarks Output
 */
let frameCount = 0;
function onHandResults(results) {
  if (!elements.canvasCtx) return;

  const ctx = elements.canvasCtx;
  ctx.save();
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

  // Draw incoming video frame
  ctx.drawImage(results.image, 0, 0, elements.canvas.width, elements.canvas.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const hand = results.multiHandLandmarks[0];

    // Draw skeleton mesh
    drawConnectors(ctx, hand, HAND_CONNECTIONS, { color: '#06b6d4', lineWidth: 3 });
    drawLandmarks(ctx, hand, { color: '#10b981', lineWidth: 2, radius: 4 });

    // Flatten landmarks to 63 floats (21 x 3)
    const flatLandmarks = [];
    for (const pt of hand) {
      flatLandmarks.push(pt.x, pt.y, pt.z);
    }

    elements.trackingStatusText.textContent = 'Hand detected • Extracting PSL features';
    elements.metricLandmarks.textContent = '21 points (Active)';

    frameCount = (frameCount + 1) % 30;
    elements.metricBuffer.textContent = `${frameCount + 1} / 30 frames`;

    // Stream to backend WebSocket
    if (state.webSocket && state.webSocket.readyState === WebSocket.OPEN) {
      state.webSocket.send(JSON.stringify({
        session_id: 'browser-webcam',
        landmarks: flatLandmarks
      }));
    }
  } else {
    elements.trackingStatusText.textContent = 'Position hand inside frame';
    elements.metricLandmarks.textContent = 'Searching...';
  }

  ctx.restore();
}

/**
 * Process Prediction from Backend
 */
function handlePredictionResult(data) {
  // Update showcase
  elements.detectedUrdu.textContent = data.urdu || 'الف';
  elements.detectedEnglish.textContent = data.english || 'Alif';
  elements.detectedPhonetic.textContent = `Phonetic: /${data.phonetic || data.english}/`;

  const confPercent = (data.confidence * 100).toFixed(1);
  elements.confidencePercentage.textContent = `${confPercent}%`;
  elements.confidenceBar.style.width = `${confPercent}%`;

  if (elements.signCategoryBadge) {
    elements.signCategoryBadge.textContent =
      data.category === 'letter' ? 'Urdu Letter' :
      data.category === 'digit' ? 'Number' : 'Phrase';
  }

  // Trigger TTS voice output
  speakText(data.urdu, data.phonetic);

  // Append to history
  addToHistory(data);
}

function addToHistory(item) {
  // Prevent immediate duplicates within 2 seconds
  const last = state.recentDetections[0];
  if (last && last.english === item.english && (Date.now() - last._t < 2500)) {
    return;
  }

  item._t = Date.now();
  item.timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  state.recentDetections.unshift(item);
  if (state.recentDetections.length > 25) state.recentDetections.pop();

  state.sentenceTokens.push(item);
  if (state.sentenceTokens.length > 15) state.sentenceTokens.shift();

  renderHistory();
  renderSentence();
}

function renderHistory() {
  if (!elements.historyList) return;
  elements.historyList.innerHTML = '';
  elements.historyCount.textContent = `${state.recentDetections.length} items`;

  state.recentDetections.slice(0, 7).forEach(det => {
    const row = document.createElement('div');
    row.className = 'history-item';
    row.innerHTML = `
      <div class="history-item-left">
        <span class="history-urdu">${det.urdu}</span>
        <div>
          <span class="history-english">${det.english}</span>
          <span class="history-time">• ${det.timeStr}</span>
        </div>
      </div>
      <span class="history-badge">${(det.confidence * 100).toFixed(0)}%</span>
    `;
    elements.historyList.appendChild(row);
  });
}

function renderSentence() {
  if (!elements.sentenceContent) return;
  if (state.sentenceTokens.length === 0) {
    elements.sentenceContent.innerHTML = '<span class="empty-prompt">Signs you perform will automatically appear here in Urdu script...</span>';
    return;
  }

  const urduWords = state.sentenceTokens.map(t => t.urdu).join(' ');
  elements.sentenceContent.textContent = urduWords;
}

/**
 * Quick Test Sign Feature
 */
function setupQuickTest() {
  if (!elements.btnQuickSample) return;

  elements.btnQuickSample.addEventListener('click', async () => {
    if (state.allSigns.length === 0) {
      await loadAllSigns();
    }

    // Pick a random PSL sign
    const sign = state.allSigns[Math.floor(Math.random() * state.allSigns.length)];
    const mockConfidence = 0.92 + Math.random() * 0.07;

    // Send mock landmark sequence to backend via POST or simulate result
    handlePredictionResult({
      status: 'success',
      sign_key: sign.id,
      english: sign.english,
      urdu: sign.urdu,
      phonetic: sign.phonetic,
      category: sign.category,
      confidence: mockConfidence
    });

    showToast(`Quick Tested Sign: "${sign.english}" (${sign.urdu})`);
  });
}

/**
 * Load Vocabulary of 51 PSL Signs
 */
async function loadAllSigns() {
  try {
    const res = await fetch('/api/detection/signs');
    if (res.ok) {
      state.allSigns = await res.json();
      renderSignsGrid(state.allSigns);
    }
  } catch (err) {
    console.error('Error loading signs:', err);
  }
}

function renderSignsGrid(signs) {
  if (!elements.signsGrid) return;
  elements.signsGrid.innerHTML = '';

  signs.forEach(sign => {
    const card = document.createElement('div');
    card.className = 'sign-card';
    card.innerHTML = `
      <div class="sign-card-urdu">${sign.urdu}</div>
      <div class="sign-card-english">${sign.english}</div>
      <div class="sign-card-phonetic">/${sign.phonetic}/</div>
      <div class="sign-card-actions">
        <button class="btn-listen" title="Pronounce sign">🔊 Listen</button>
        <button class="btn-test" title="Send to real-time detector">⚡ Test</button>
      </div>
    `;

    card.querySelector('.btn-listen').addEventListener('click', () => {
      speakText(sign.urdu, sign.phonetic, true);
    });

    card.querySelector('.btn-test').addEventListener('click', () => {
      handlePredictionResult({
        status: 'success',
        sign_key: sign.id,
        english: sign.english,
        urdu: sign.urdu,
        phonetic: sign.phonetic,
        category: sign.category,
        confidence: 0.95
      });
      // Switch to detect view to see results
      document.getElementById('tab-detect')?.click();
      showToast(`Selected sign "${sign.english}" sent to detector!`);
    });

    elements.signsGrid.appendChild(card);
  });
}

function setupDictionaryFilters() {
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.getAttribute('data-filter');
      if (filter === 'all') {
        renderSignsGrid(state.allSigns);
      } else {
        const filtered = state.allSigns.filter(s => s.category === filter);
        renderSignsGrid(filtered);
      }
    });
  });
}

/**
 * Analytics Dashboard Data Fetching & Visualization
 */
async function loadAnalytics() {
  try {
    const res = await fetch('/api/analytics/summary');
    if (!res.ok) return;

    const data = await res.json();

    // KPIs
    document.getElementById('kpi-total').textContent = data.total_detections.toLocaleString();
    document.getElementById('kpi-today').textContent = data.today_detections.toLocaleString();
    document.getElementById('kpi-conf').textContent = `${(data.avg_confidence * 100).toFixed(1)}%`;

    // Render Trend SVG Chart
    renderTrendChart(data.daily_trend || []);

    // Render Category Breakdown
    renderCategoryBreakdown(data);

    // Top Signs
    renderTopSigns(data.popular_signs || []);

    // Table
    renderAnalyticsTable(data.recent_activity || []);
  } catch (err) {
    console.error('Analytics load error:', err);
  }
}

function renderTrendChart(trend) {
  const container = document.getElementById('trend-chart-container');
  if (!container || trend.length === 0) return;

  const maxVal = Math.max(...trend.map(t => t.detections), 10);
  const width = container.clientWidth || 500;
  const height = 220;
  const padding = 35;

  const barWidth = (width - padding * 2) / trend.length - 12;

  let barsSvg = '';
  trend.forEach((d, idx) => {
    const x = padding + idx * (barWidth + 12);
    const barHeight = (d.detections / maxVal) * (height - padding * 2);
    const y = height - padding - barHeight;

    barsSvg += `
      <g class="bar-group">
        <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="6"
              fill="url(#trendGradient)" opacity="0.85">
          <title>${d.date}: ${d.detections} detections (avg ${(d.avg_confidence * 100).toFixed(0)}%)</title>
        </rect>
        <text x="${x + barWidth / 2}" y="${height - 12}" fill="#94a3b8" font-size="11" text-anchor="middle">
          ${d.date}
        </text>
        <text x="${x + barWidth / 2}" y="${y - 6}" fill="#f8fafc" font-size="10" font-weight="600" text-anchor="middle">
          ${d.detections}
        </text>
      </g>
    `;
  });

  container.innerHTML = `
    <svg width="100%" height="100%" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <defs>
        <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#10b981" />
          <stop offset="100%" stop-color="#06b6d4" stop-opacity="0.2" />
        </linearGradient>
      </defs>
      <!-- Background Grid lines -->
      <line x1="${padding}" y1="${padding}" x2="${width - padding}" y2="${padding}" stroke="rgba(255,255,255,0.05)" />
      <line x1="${padding}" y1="${height / 2}" x2="${width - padding}" y2="${height / 2}" stroke="rgba(255,255,255,0.05)" />
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="rgba(255,255,255,0.1)" />
      ${barsSvg}
    </svg>
  `;
}

function renderCategoryBreakdown(data) {
  const container = document.getElementById('category-breakdown-container');
  if (!container) return;

  const total = Math.max(1, data.letters_count + data.digits_count + data.phrases_count);
  const pLetters = Math.round((data.letters_count / total) * 100);
  const pDigits = Math.round((data.digits_count / total) * 100);
  const pPhrases = Math.round((data.phrases_count / total) * 100);

  container.innerHTML = `
    <div class="breakdown-bar-item">
      <div class="breakdown-info">
        <span>Urdu Letters (حروفِ تہجی)</span>
        <strong>${data.letters_count} (${pLetters}%)</strong>
      </div>
      <div class="breakdown-track">
        <div class="breakdown-fill" style="width: ${pLetters}%; background: #10b981;"></div>
      </div>
    </div>

    <div class="breakdown-bar-item">
      <div class="breakdown-info">
        <span>Numbers / Digits (اعداد)</span>
        <strong>${data.digits_count} (${pDigits}%)</strong>
      </div>
      <div class="breakdown-track">
        <div class="breakdown-fill" style="width: ${pDigits}%; background: #06b6d4;"></div>
      </div>
    </div>

    <div class="breakdown-bar-item">
      <div class="breakdown-info">
        <span>Conversational Phrases (جملے)</span>
        <strong>${data.phrases_count} (${pPhrases}%)</strong>
      </div>
      <div class="breakdown-track">
        <div class="breakdown-fill" style="width: ${pPhrases}%; background: #8b5cf6;"></div>
      </div>
    </div>
  `;
}

function renderTopSigns(popularSigns) {
  const container = document.getElementById('top-signs-list');
  if (!container) return;
  container.innerHTML = '';

  popularSigns.slice(0, 6).forEach((s, idx) => {
    const row = document.createElement('div');
    row.className = 'top-sign-row';
    row.innerHTML = `
      <div class="top-sign-left">
        <span class="top-sign-rank">#${idx + 1}</span>
        <span class="top-sign-urdu">${s.urdu}</span>
        <div>
          <span style="font-weight: 600;">${s.english}</span>
          <div style="font-size: 0.72rem; color: #94a3b8;">${s.category} • ${(s.avg_confidence * 100).toFixed(0)}% conf</div>
        </div>
      </div>
      <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">${s.count} logs</span>
    `;
    container.appendChild(row);
  });
}

function renderAnalyticsTable(logs) {
  const tbody = document.getElementById('analytics-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  logs.forEach(log => {
    const tr = document.createElement('tr');
    const dateStr = new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    tr.innerHTML = `
      <td class="table-urdu">${log.urdu_label}</td>
      <td><strong>${log.english_label}</strong></td>
      <td><span class="badge">${log.category || 'letter'}</span></td>
      <td><span style="color: #34d399; font-weight: 600;">${(log.confidence * 100).toFixed(1)}%</span></td>
      <td style="color: #94a3b8;">${dateStr}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Start application when DOM is ready
window.addEventListener('DOMContentLoaded', initApp);
