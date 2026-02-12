"""批次比對報告的 HTML 結構"""

HTML_BODY = """
<header class="header">
    <div class="logo">
        <div class="logo-icon">BG</div>
        <div class="logo-text">Compare <span>v1.0</span></div>
    </div>
    <div class="header-actions">
        <span style="font-size: 12px; color: var(--text-muted);">快捷鍵：</span>
        <kbd>\\u2190</kbd><kbd>\\u2192</kbd> 切換圖片
        <kbd>1</kbd>-<kbd>5</kbd> 切換配置
    </div>
</header>

<main class="main">
    <aside class="sidebar">
        <section>
            <h3 class="section-title">測試圖片</h3>
            <div class="image-list" id="imageList"></div>
        </section>

        <section>
            <h3 class="section-title">處理配置</h3>
            <div class="config-list" id="configList"></div>
        </section>

        <section>
            <h3 class="section-title">品質分析</h3>
            <div class="stats-panel">
                <div class="stats-grid" id="statsGrid"></div>
            </div>
        </section>
    </aside>

    <div class="compare-area">
        <div class="compare-header">
            <div class="compare-title">
                <strong id="currentImageName">選擇圖片</strong>
                <span id="currentConfigName"></span>
            </div>
            <div class="view-toggle">
                <button class="view-btn active" data-view="slider">滑動比對</button>
                <button class="view-btn" data-view="sidebyside">並排顯示</button>
            </div>
        </div>

        <div class="compare-container">
            <div class="checker-bg"></div>

            <div class="slider-compare" id="sliderView">
                <div class="slider-wrapper" id="sliderWrapper">
                    <img class="slider-image processed" id="processedImage" src="" alt="Processed">
                    <img class="slider-image original" id="originalImage" src="" alt="Original">
                    <div class="slider-divider" id="sliderDivider"></div>
                </div>
                <div class="slider-label left">原圖</div>
                <div class="slider-label right">處理後</div>
            </div>

            <div class="side-by-side hidden" id="sideBySideView">
                <div class="side-panel original">
                    <div class="checker-bg"></div>
                    <img id="sideOriginal" src="" alt="Original">
                    <div class="panel-label">原圖</div>
                </div>
                <div class="side-panel processed">
                    <div class="checker-bg"></div>
                    <img id="sideProcessed" src="" alt="Processed">
                    <div class="panel-label">處理後</div>
                </div>
            </div>
        </div>
    </div>
</main>

<div class="keyboard-hints">
    <div class="hint-item"><kbd>V</kbd> 切換視圖</div>
    <div class="hint-item"><kbd>Space</kbd> 顯示/隱藏原圖</div>
</div>
"""
