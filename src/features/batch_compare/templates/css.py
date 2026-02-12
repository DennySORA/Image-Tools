"""批次比對報告的 CSS 樣式"""

CSS_STYLES = """
:root {
    --bg-primary: #0a0a0b;
    --bg-secondary: #141416;
    --bg-tertiary: #1c1c1f;
    --bg-elevated: #232326;
    --text-primary: #f5f5f7;
    --text-secondary: #a1a1a6;
    --text-muted: #6e6e73;
    --accent-cyan: #00d4ff;
    --accent-green: #32d74b;
    --accent-orange: #ff9f0a;
    --accent-red: #ff453a;
    --accent-purple: #bf5af2;
    --border-color: #2d2d30;
    --checker-light: #3a3a3d;
    --checker-dark: #2a2a2d;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --font-mono: 'JetBrains Mono', monospace;
    --font-sans: 'Noto Sans TC', -apple-system, BlinkMacSystemFont, sans-serif;
    --shadow-lg: 0 24px 48px rgba(0, 0, 0, 0.4);
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth: 300ms cubic-bezier(0.4, 0, 0.2, 1);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: var(--font-sans);
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
}

.header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    background: rgba(10, 10, 11, 0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-color);
    padding: 0 24px; height: 56px;
    display: flex; align-items: center; justify-content: space-between;
}

.logo { display: flex; align-items: center; gap: 12px; }

.logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono); font-weight: 700; font-size: 14px;
}

.logo-text {
    font-family: var(--font-mono); font-weight: 600;
    font-size: 14px; letter-spacing: -0.5px;
}
.logo-text span { color: var(--text-muted); font-weight: 400; }

.header-actions { display: flex; align-items: center; gap: 12px; }

.kbd {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 24px; height: 22px; padding: 0 6px;
    background: var(--bg-tertiary); border: 1px solid var(--border-color);
    border-radius: 4px; font-family: var(--font-mono);
    font-size: 11px; color: var(--text-muted);
}

.main {
    padding-top: 56px;
    display: grid; grid-template-columns: 280px 1fr;
    min-height: 100vh;
}

.sidebar {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    padding: 20px 16px;
    display: flex; flex-direction: column; gap: 24px;
    overflow-y: auto; height: calc(100vh - 56px);
    position: sticky; top: 56px;
}

.section-title {
    font-family: var(--font-mono); font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--text-muted); margin-bottom: 12px;
}

.image-list { display: flex; flex-direction: column; gap: 4px; }

.image-item {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 12px; border-radius: var(--radius-md);
    cursor: pointer; transition: all var(--transition-fast);
    border: 1px solid transparent;
}
.image-item:hover { background: var(--bg-tertiary); }
.image-item.active {
    background: var(--bg-elevated);
    border-color: var(--accent-cyan);
    box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.2);
}

.image-thumb {
    width: 40px; height: 40px;
    border-radius: var(--radius-sm); object-fit: cover;
    background: var(--checker-dark);
    background-image:
        linear-gradient(45deg, var(--checker-light) 25%, transparent 25%),
        linear-gradient(-45deg, var(--checker-light) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, var(--checker-light) 75%),
        linear-gradient(-45deg, transparent 75%, var(--checker-light) 75%);
    background-size: 8px 8px;
    background-position: 0 0, 0 4px, 4px -4px, -4px 0px;
}

.image-info { flex: 1; min-width: 0; }
.image-name {
    font-size: 13px; font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.image-score {
    font-family: var(--font-mono); font-size: 11px;
    color: var(--text-secondary);
}

.config-list { display: flex; flex-direction: column; gap: 6px; }

.config-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px; border-radius: var(--radius-md);
    cursor: pointer; transition: all var(--transition-fast);
    border: 1px solid transparent; background: var(--bg-tertiary);
}
.config-item:hover { background: var(--bg-elevated); }
.config-item.active {
    border-color: var(--accent-green);
    box-shadow: 0 0 0 1px rgba(50, 215, 75, 0.2);
}
.config-item.active .config-name { color: var(--accent-green); }

.config-info { display: flex; flex-direction: column; gap: 4px; }
.config-name {
    font-family: var(--font-mono); font-size: 12px; font-weight: 600;
    transition: color var(--transition-fast);
}
.config-desc { font-size: 11px; color: var(--text-muted); }
.config-score { display: flex; align-items: center; gap: 6px; }

.score-badge {
    font-family: var(--font-mono); font-size: 13px; font-weight: 700;
    padding: 4px 8px; border-radius: var(--radius-sm);
    background: rgba(50, 215, 75, 0.15); color: var(--accent-green);
}
.score-badge.warning { background: rgba(255, 159, 10, 0.15); color: var(--accent-orange); }
.score-badge.danger { background: rgba(255, 69, 58, 0.15); color: var(--accent-red); }

.stats-panel { background: var(--bg-tertiary); border-radius: var(--radius-md); padding: 16px; }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.stat-item { display: flex; flex-direction: column; gap: 4px; }
.stat-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
.stat-value { font-family: var(--font-mono); font-size: 18px; font-weight: 700; }
.stat-value.green { color: var(--accent-green); }
.stat-value.orange { color: var(--accent-orange); }
.stat-value.red { color: var(--accent-red); }
.stat-value.cyan { color: var(--accent-cyan); }
.stat-bar { height: 4px; background: var(--bg-primary); border-radius: 2px; overflow: hidden; margin-top: 4px; }
.stat-bar-fill { height: 100%; border-radius: 2px; transition: width var(--transition-smooth); }
.stat-bar-fill.green { background: var(--accent-green); }
.stat-bar-fill.orange { background: var(--accent-orange); }
.stat-bar-fill.red { background: var(--accent-red); }
.stat-bar-fill.cyan { background: var(--accent-cyan); }

.compare-area { padding: 24px; display: flex; flex-direction: column; gap: 20px; }
.compare-header { display: flex; align-items: center; justify-content: space-between; }
.compare-title { font-family: var(--font-mono); font-size: 13px; color: var(--text-secondary); }
.compare-title strong { color: var(--text-primary); font-weight: 600; }

.view-toggle { display: flex; background: var(--bg-tertiary); border-radius: var(--radius-sm); padding: 3px; gap: 2px; }
.view-btn {
    padding: 6px 14px; border: none; background: transparent;
    color: var(--text-secondary); font-family: var(--font-mono);
    font-size: 12px; font-weight: 500; border-radius: 4px;
    cursor: pointer; transition: all var(--transition-fast);
}
.view-btn:hover { color: var(--text-primary); }
.view-btn.active { background: var(--bg-elevated); color: var(--text-primary); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); }

.compare-container {
    flex: 1; display: flex; align-items: center; justify-content: center;
    background: var(--bg-secondary); border-radius: var(--radius-lg);
    overflow: hidden; position: relative; min-height: 500px;
}

.checker-bg {
    position: absolute; inset: 0;
    background-color: var(--checker-dark);
    background-image:
        linear-gradient(45deg, var(--checker-light) 25%, transparent 25%),
        linear-gradient(-45deg, var(--checker-light) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, var(--checker-light) 75%),
        linear-gradient(-45deg, transparent 75%, var(--checker-light) 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
}

.slider-compare { position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
.slider-wrapper { position: relative; max-width: 100%; max-height: 100%; }
.slider-image { display: block; max-width: 100%; max-height: calc(100vh - 200px); object-fit: contain; }
.slider-image.original { position: absolute; top: 0; left: 0; clip-path: inset(0 0 0 0); }
.slider-image.processed { position: relative; z-index: 1; }

.slider-divider {
    position: absolute; top: 0; bottom: 0; width: 3px;
    background: var(--text-primary); cursor: ew-resize; z-index: 10;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
}
.slider-divider::before {
    content: ''; position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%); width: 40px; height: 40px;
    background: var(--bg-elevated); border: 2px solid var(--text-primary);
    border-radius: 50%; box-shadow: var(--shadow-lg);
}
.slider-divider::after {
    content: '\\2194'; position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%); font-size: 16px;
    color: var(--text-primary); z-index: 1;
}

.slider-label {
    position: absolute; bottom: 16px; padding: 6px 12px;
    background: rgba(10, 10, 11, 0.85); backdrop-filter: blur(10px);
    border-radius: var(--radius-sm); font-family: var(--font-mono);
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
    text-transform: uppercase; z-index: 5;
}
.slider-label.left { left: 16px; color: var(--accent-orange); }
.slider-label.right { right: 16px; color: var(--accent-green); }

.side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; width: 100%; height: 100%; padding: 4px; }
.side-panel { position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: var(--radius-md); }
.side-panel img { max-width: 100%; max-height: calc(100vh - 200px); object-fit: contain; }
.side-panel .panel-label {
    position: absolute; top: 12px; left: 12px; padding: 6px 12px;
    background: rgba(10, 10, 11, 0.85); backdrop-filter: blur(10px);
    border-radius: var(--radius-sm); font-family: var(--font-mono);
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase;
}
.side-panel.original .panel-label { color: var(--accent-orange); }
.side-panel.processed .panel-label { color: var(--accent-green); }

.hidden { display: none !important; }

.keyboard-hints {
    position: fixed; bottom: 20px; right: 20px;
    display: flex; gap: 12px; padding: 12px 16px;
    background: rgba(20, 20, 22, 0.9); backdrop-filter: blur(10px);
    border: 1px solid var(--border-color); border-radius: var(--radius-md);
    font-size: 12px; color: var(--text-secondary); z-index: 50;
}
.hint-item { display: flex; align-items: center; gap: 6px; }

.loading {
    display: flex; align-items: center; justify-content: center; gap: 12px;
    color: var(--text-muted); font-family: var(--font-mono); font-size: 13px;
}
.loading::before {
    content: ''; width: 20px; height: 20px;
    border: 2px solid var(--border-color); border-top-color: var(--accent-cyan);
    border-radius: 50%; animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 900px) {
    .main { grid-template-columns: 1fr; }
    .sidebar { display: none; }
    .compare-area { padding: 16px; }
}
"""
