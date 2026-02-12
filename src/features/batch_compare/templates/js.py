"""批次比對報告的 JavaScript 程式碼"""


def get_javascript(
    test_images_json: str, configs_json: str, best_config: str, input_rel_path: str
) -> str:
    """
    生成 JavaScript 程式碼

    Args:
        test_images_json: 測試圖片 JSON 字串
        configs_json: 配置 JSON 字串
        best_config: 最佳配置名稱
        input_rel_path: 輸入目錄相對路徑

    Returns:
        完整的 JavaScript 程式碼
    """
    return f'''
const reportData = {{
    test_images: {test_images_json},
    configs: {configs_json},
    best_config: "{best_config}"
}};
const inputRelPath = "{input_rel_path}";

let fullReport = null;
fetch('test_report.json')
    .then(r => r.json())
    .then(data => {{ fullReport = data; updateStats(); }})
    .catch(e => console.log('Could not load detailed report'));

let currentImage = reportData.test_images[0];
let currentConfig = reportData.best_config || Object.keys(reportData.configs)[0];
let currentView = 'slider';
let sliderPosition = 50;

const imageList = document.getElementById('imageList');
const configList = document.getElementById('configList');
const statsGrid = document.getElementById('statsGrid');
const currentImageName = document.getElementById('currentImageName');
const currentConfigName = document.getElementById('currentConfigName');
const sliderView = document.getElementById('sliderView');
const sideBySideView = document.getElementById('sideBySideView');
const originalImage = document.getElementById('originalImage');
const processedImage = document.getElementById('processedImage');
const sideOriginal = document.getElementById('sideOriginal');
const sideProcessed = document.getElementById('sideProcessed');
const sliderDivider = document.getElementById('sliderDivider');
const sliderWrapper = document.getElementById('sliderWrapper');

function init() {{
    renderImageList(); renderConfigList(); updateImages();
    setupSlider(); setupKeyboard(); setupViewToggle();
}}

function renderImageList() {{
    imageList.innerHTML = reportData.test_images.map((img, i) => {{
        const shortName = img.replace('Gemini_Generated_Image_', '').replace('.png', '');
        const displayName = shortName.length > 15 ? shortName.slice(0, 12) + '...' : shortName;
        return `<div class="image-item ${{img === currentImage ? 'active' : ''}}" data-image="${{img}}">
            <img class="image-thumb" src="${{inputRelPath}}/${{img}}" alt="${{shortName}}" onerror="this.style.display='none'">
            <div class="image-info"><div class="image-name">${{displayName}}</div><div class="image-score">#${{i + 1}}</div></div>
        </div>`;
    }}).join('');
    imageList.querySelectorAll('.image-item').forEach(item => {{
        item.addEventListener('click', () => {{ currentImage = item.dataset.image; updateSelection(); }});
    }});
}}

function renderConfigList() {{
    const configs = Object.entries(reportData.configs);
    configList.innerHTML = configs.map(([key, config], i) => {{
        const scoreClass = config.average_score >= 95 ? '' : config.average_score >= 90 ? 'warning' : 'danger';
        const isBest = key === reportData.best_config;
        return `<div class="config-item ${{key === currentConfig ? 'active' : ''}}" data-config="${{key}}">
            <div class="config-info"><div class="config-name">${{isBest ? '\\uD83C\\uDFC6 ' : ''}}${{key}}</div><div class="config-desc">${{config.description}}</div></div>
            <div class="config-score"><span class="score-badge ${{scoreClass}}">${{config.average_score.toFixed(1)}}</span></div>
        </div>`;
    }}).join('');
    configList.querySelectorAll('.config-item').forEach(item => {{
        item.addEventListener('click', () => {{ currentConfig = item.dataset.config; updateSelection(); }});
    }});
}}

function updateSelection() {{
    imageList.querySelectorAll('.image-item').forEach(item => {{ item.classList.toggle('active', item.dataset.image === currentImage); }});
    configList.querySelectorAll('.config-item').forEach(item => {{ item.classList.toggle('active', item.dataset.config === currentConfig); }});
    updateImages(); updateStats();
}}

function updateImages() {{
    const originalPath = `${{inputRelPath}}/${{currentImage}}`;
    const processedPath = `${{currentConfig}}/${{currentImage}}`;
    originalImage.src = originalPath; processedImage.src = processedPath;
    sideOriginal.src = originalPath; sideProcessed.src = processedPath;
    const shortName = currentImage.replace('Gemini_Generated_Image_', '').replace('.png', '');
    currentImageName.textContent = shortName;
    currentConfigName.textContent = ` \\u2192 ${{currentConfig}}`;
}}

function updateStats() {{
    if (!fullReport) {{ statsGrid.innerHTML = '<div class="loading">載入中...</div>'; return; }}
    const configData = fullReport.configs[currentConfig];
    const imageData = configData?.images?.[currentImage];
    if (!imageData) {{ statsGrid.innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">無數據</div>'; return; }}
    const stats = [
        {{ label: '綠幕殘留', value: imageData.green_spill.score, color: imageData.green_spill.score >= 90 ? 'green' : 'orange' }},
        {{ label: '邊緣溢出', value: imageData.edge_bleeding.score, color: imageData.edge_bleeding.score >= 90 ? 'green' : 'orange' }},
        {{ label: '過度移除', value: imageData.over_removal.score, color: imageData.over_removal.score >= 50 ? 'green' : 'red' }},
        {{ label: '總分', value: imageData.total_score, color: 'cyan' }}
    ];
    statsGrid.innerHTML = stats.map(stat => `<div class="stat-item">
        <div class="stat-label">${{stat.label}}</div><div class="stat-value ${{stat.color}}">${{stat.value.toFixed(1)}}</div>
        <div class="stat-bar"><div class="stat-bar-fill ${{stat.color}}" style="width: ${{stat.value}}%"></div></div>
    </div>`).join('');
}}

function setupSlider() {{
    let isDragging = false;
    const updateSlider = (clientX) => {{
        const rect = sliderWrapper.getBoundingClientRect();
        const x = clientX - rect.left;
        const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
        sliderPosition = percent;
        sliderDivider.style.left = `${{percent}}%`;
        originalImage.style.clipPath = `inset(0 ${{100 - percent}}% 0 0)`;
    }};
    sliderDivider.addEventListener('mousedown', (e) => {{ isDragging = true; e.preventDefault(); }});
    document.addEventListener('mousemove', (e) => {{ if (isDragging) updateSlider(e.clientX); }});
    document.addEventListener('mouseup', () => {{ isDragging = false; }});
    sliderWrapper.addEventListener('click', (e) => {{ if (!isDragging) updateSlider(e.clientX); }});
    sliderDivider.style.left = '50%';
    originalImage.style.clipPath = 'inset(0 50% 0 0)';
}}

function setupKeyboard() {{
    document.addEventListener('keydown', (e) => {{
        const images = reportData.test_images;
        const configs = Object.keys(reportData.configs);
        const currentImageIndex = images.indexOf(currentImage);
        switch(e.key) {{
            case 'ArrowLeft':
                if (currentImageIndex > 0) {{ currentImage = images[currentImageIndex - 1]; updateSelection(); }}
                break;
            case 'ArrowRight':
                if (currentImageIndex < images.length - 1) {{ currentImage = images[currentImageIndex + 1]; updateSelection(); }}
                break;
            case '1': case '2': case '3': case '4': case '5':
                const idx = parseInt(e.key) - 1;
                if (idx < configs.length) {{ currentConfig = configs[idx]; updateSelection(); }}
                break;
            case 'v': case 'V': toggleView(); break;
            case ' ':
                e.preventDefault();
                if (currentView === 'slider') {{
                    if (originalImage.style.clipPath === 'inset(0 0% 0 0)') {{
                        originalImage.style.clipPath = 'inset(0 50% 0 0)'; sliderDivider.style.left = '50%';
                    }} else {{
                        originalImage.style.clipPath = 'inset(0 0% 0 0)'; sliderDivider.style.left = '100%';
                    }}
                }}
                break;
        }}
    }});
}}

function setupViewToggle() {{
    document.querySelectorAll('.view-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{ setView(btn.dataset.view); }});
    }});
}}

function setView(view) {{
    currentView = view;
    document.querySelectorAll('.view-btn').forEach(btn => {{ btn.classList.toggle('active', btn.dataset.view === view); }});
    sliderView.classList.toggle('hidden', view !== 'slider');
    sideBySideView.classList.toggle('hidden', view !== 'sidebyside');
}}

function toggleView() {{ setView(currentView === 'slider' ? 'sidebyside' : 'slider'); }}

init();
'''
