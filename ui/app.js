/* ============================================================
   JARVIS — Frontend Application Logic
   WebSocket connection, chat management, avatar animation
   ============================================================ */

const WS_URL = `ws://${window.location.host}/ws`;

// ---- STATE ----
let ws = null;
let isConnected = false;
let currentState = 'idle'; // idle | thinking | speaking | executing

// ---- DOM REFS ----
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const avatarCanvas = document.getElementById('avatarCanvas');
const avatarMode = document.getElementById('avatarMode');
const feedItems = document.getElementById('feedItems');

// ============================================================
// WEBSOCKET CONNECTION
// ============================================================

function connectWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        isConnected = true;
        setStatus('online', 'Online');
        console.log('[WS] Connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };

    ws.onclose = () => {
        isConnected = false;
        setStatus('error', 'Disconnected');
        console.log('[WS] Disconnected. Reconnecting in 3s...');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('[WS] Error:', err);
        setStatus('error', 'Connection Error');
    };
}

// ============================================================
// MESSAGE HANDLING
// ============================================================

function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || !isConnected) return;

    // Add user message to chat
    addMessage('user', text);
    chatInput.value = '';
    autoResizeTextarea();

    // Show thinking indicator
    showThinking();
    setStatus('thinking', 'Thinking...');

    // Send to server
    ws.send(JSON.stringify({ type: 'message', content: text }));
    sendBtn.disabled = true;
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'response':
            removeThinking();
            if (data.speak) {
                addMessage('bot', data.speak);
            }
            if (data.actions_results && data.actions_results.length > 0) {
                addToolResults(data.actions_results);
            }
            setStatus('online', 'Online');
            setAvatarState('idle');
            sendBtn.disabled = false;
            break;

        case 'thinking':
            setStatus('thinking', 'Thinking...');
            setAvatarState('thinking');
            break;

        case 'speaking':
            setStatus('speaking', 'Speaking...');
            setAvatarState('speaking');
            break;

        case 'executing':
            setStatus('thinking', `Running: ${data.tool || '...'}`);
            setAvatarState('executing');
            addFeedItem(data.tool, 'running', `Executing ${data.tool}...`);
            break;

        case 'tool_result':
            const status = data.result?.status === 'ok' ? 'success' : 'error';
            addFeedItem(data.tool, status, data.result?.result || 'Done');
            break;

        case 'error':
            removeThinking();
            addMessage('bot', `Error: ${data.message}`);
            setStatus('error', 'Error');
            sendBtn.disabled = false;
            break;

        case 'startup':
            removeWelcome();
            addMessage('bot', data.speak || 'JARVIS Online.');
            setStatus('online', 'Online');
            setAvatarState('idle');
            break;
    }
}

// ============================================================
// CHAT UI FUNCTIONS
// ============================================================

function addMessage(role, text) {
    removeWelcome();
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    
    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = role === 'user' ? 'You' : 'SATHI';
    
    const body = document.createElement('div');
    body.className = 'message-body';
    body.textContent = text;
    
    msg.appendChild(label);
    msg.appendChild(body);
    chatMessages.appendChild(msg);
    scrollToBottom();
}

function addToolResults(results) {
    const lastMsg = chatMessages.querySelector('.message.bot:last-of-type .message-body');
    if (!lastMsg) return;

    results.forEach(r => {
        const div = document.createElement('div');
        div.className = `tool-result ${r.status === 'error' ? 'error' : ''}`;
        const resultText = typeof r.result === 'object' ? JSON.stringify(r.result, null, 2) : r.result;
        div.textContent = `[${r.tool}] ${resultText}`;
        lastMsg.appendChild(div);
    });
    scrollToBottom();
}

function showThinking() {
    const indicator = document.createElement('div');
    indicator.className = 'thinking-indicator';
    indicator.id = 'thinkingIndicator';
    indicator.innerHTML = `
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        <span>SATHI is thinking...</span>
    `;
    chatMessages.appendChild(indicator);
    scrollToBottom();
}

function removeThinking() {
    const el = document.getElementById('thinkingIndicator');
    if (el) el.remove();
}

function removeWelcome() {
    const el = chatMessages.querySelector('.welcome-msg');
    if (el) el.remove();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

// ============================================================
// STATUS & FEED
// ============================================================

function setStatus(state, text) {
    statusIndicator.className = `status-indicator ${state}`;
    statusText.textContent = text;
}

function setAvatarState(state) {
    currentState = state;
    const labels = { idle: 'Idle', thinking: 'Thinking', speaking: 'Speaking', executing: 'Executing' };
    avatarMode.textContent = labels[state] || 'Idle';
}

function addFeedItem(tool, status, text) {
    const icons = { running: '⏳', success: '✓', error: '✗' };
    const item = document.createElement('div');
    item.className = `feed-item ${status}`;
    item.innerHTML = `<span class="feed-icon">${icons[status] || '•'}</span>${tool}: ${typeof text === 'string' ? text.substring(0, 60) : 'Done'}`;
    feedItems.insertBefore(item, feedItems.firstChild);

    // Keep max 20 items
    while (feedItems.children.length > 20) {
        feedItems.removeChild(feedItems.lastChild);
    }
}

// ============================================================
// INPUT HANDLING
// ============================================================

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

chatInput.addEventListener('input', autoResizeTextarea);
sendBtn.addEventListener('click', sendMessage);

function autoResizeTextarea() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}

// ============================================================
// AVATAR CANVAS ANIMATION
// ============================================================

const ctx = avatarCanvas.getContext('2d');
const W = avatarCanvas.width;
const H = avatarCanvas.height;
const centerX = W / 2;
const centerY = H / 2;

// Particle system for holographic avatar
const particles = [];
const PARTICLE_COUNT = 120;

class Particle {
    constructor() {
        this.reset();
    }

    reset() {
        const angle = Math.random() * Math.PI * 2;
        const radius = 40 + Math.random() * 100;
        this.x = centerX + Math.cos(angle) * radius;
        this.y = centerY + Math.sin(angle) * radius;
        this.baseX = this.x;
        this.baseY = this.y;
        this.size = Math.random() * 2.5 + 0.5;
        this.speedX = (Math.random() - 0.5) * 0.3;
        this.speedY = (Math.random() - 0.5) * 0.3;
        this.opacity = Math.random() * 0.6 + 0.2;
        this.hue = 180 + Math.random() * 40; // cyan range
        this.phase = Math.random() * Math.PI * 2;
    }

    update(time) {
        const breathe = Math.sin(time * 0.001 + this.phase) * 4;
        const dx = this.x - centerX;
        const dy = this.y - centerY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx);

        if (currentState === 'speaking') {
            // Pulsing outward when speaking
            const pulse = Math.sin(time * 0.008 + this.phase) * 15;
            this.x = centerX + Math.cos(angle) * (dist + pulse * 0.3);
            this.y = centerY + Math.sin(angle) * (dist + pulse * 0.3);
            this.opacity = 0.4 + Math.sin(time * 0.006 + this.phase) * 0.4;
        } else if (currentState === 'thinking') {
            // Orbiting when thinking
            const orbit = time * 0.002 + this.phase;
            this.x = centerX + Math.cos(angle + orbit * 0.1) * dist;
            this.y = centerY + Math.sin(angle + orbit * 0.1) * dist;
            this.opacity = 0.3 + Math.sin(time * 0.004 + this.phase) * 0.3;
        } else {
            // Gentle drift when idle
            this.x = this.baseX + Math.sin(time * 0.001 + this.phase) * 3 + breathe * 0.5;
            this.y = this.baseY + Math.cos(time * 0.0012 + this.phase) * 3;
            this.opacity = 0.2 + Math.sin(time * 0.002 + this.phase) * 0.15;
        }
    }

    draw(ctx) {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${this.hue}, 90%, 65%, ${this.opacity})`;
        ctx.fill();
    }
}

// Initialize particles
for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push(new Particle());
}

// Ring parameters
let ringPhase = 0;

function drawAvatar(time) {
    ctx.clearRect(0, 0, W, H);

    // Background glow
    const glowGrad = ctx.createRadialGradient(centerX, centerY, 20, centerX, centerY, 160);
    glowGrad.addColorStop(0, currentState === 'speaking' ? 'rgba(0, 220, 255, 0.12)' : 'rgba(0, 220, 255, 0.05)');
    glowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = glowGrad;
    ctx.fillRect(0, 0, W, H);

    // Concentric rings
    const ringCount = 4;
    for (let i = 0; i < ringCount; i++) {
        const radius = 50 + i * 28;
        const ringOpacity = currentState === 'speaking'
            ? 0.15 + Math.sin(time * 0.005 + i) * 0.1
            : 0.06 + Math.sin(time * 0.002 + i) * 0.03;
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 220, 255, ${ringOpacity})`;
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Waveform ring (main visual)
    const waveRadius = 70;
    const wavePoints = 64;
    ctx.beginPath();
    for (let i = 0; i <= wavePoints; i++) {
        const angle = (i / wavePoints) * Math.PI * 2;
        let amp = 0;
        if (currentState === 'speaking') {
            amp = Math.sin(time * 0.01 + i * 0.5) * 20 + Math.sin(time * 0.007 + i * 0.3) * 10;
        } else if (currentState === 'thinking') {
            amp = Math.sin(time * 0.003 + i * 0.2) * 8;
        } else {
            amp = Math.sin(time * 0.001 + i * 0.1) * 3;
        }
        const r = waveRadius + amp;
        const x = centerX + Math.cos(angle) * r;
        const y = centerY + Math.sin(angle) * r;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    const waveGrad = ctx.createRadialGradient(centerX, centerY, 30, centerX, centerY, 120);
    waveGrad.addColorStop(0, 'rgba(0, 220, 255, 0.1)');
    waveGrad.addColorStop(1, 'rgba(124, 58, 237, 0.05)');
    ctx.fillStyle = waveGrad;
    ctx.fill();
    ctx.strokeStyle = currentState === 'speaking' ? 'rgba(0, 220, 255, 0.6)' : 'rgba(0, 220, 255, 0.2)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Particles
    particles.forEach(p => {
        p.update(time);
        p.draw(ctx);
    });

    // Center core
    const coreSize = 12 + (currentState === 'speaking' ? Math.sin(time * 0.008) * 5 : Math.sin(time * 0.002) * 2);
    const coreGrad = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, coreSize);
    coreGrad.addColorStop(0, 'rgba(0, 220, 255, 0.9)');
    coreGrad.addColorStop(0.5, 'rgba(124, 58, 237, 0.4)');
    coreGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.beginPath();
    ctx.arc(centerX, centerY, coreSize, 0, Math.PI * 2);
    ctx.fillStyle = coreGrad;
    ctx.fill();

    requestAnimationFrame(drawAvatar);
}

// Start animation
requestAnimationFrame(drawAvatar);

// ============================================================
// INITIALIZATION
// ============================================================

connectWebSocket();
