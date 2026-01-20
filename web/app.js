// HeartMuLa Web App

const API_URL = 'http://localhost:8080';

// State
let isGenerating = false;
let currentMode = 'custom'; // 'simple' or 'custom'
let currentView = 'create'; // 'create', 'home', 'studio', 'library', 'search'
let songs = []; // Songs loaded from server
let searchQuery = '';
let sortOrder = 'newest'; // 'newest', 'oldest', 'name'
let progressPollInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadSongsFromServer();
    await checkGenerationStatus();
    initSliders();
    initModeToggle();
    initNavigation();
    initSearch();
    initWorkspaceControls();
    initAudioPlayer();
    renderWorkspaceList();
    startArtStatusPolling();
});

// Load songs from server (outputs folder)
async function loadSongsFromServer() {
    try {
        const response = await fetch(`${API_URL}/songs`);
        if (response.ok) {
            const data = await response.json();
            songs = data.songs.map(s => ({
                filename: s.filename,
                title: s.title,
                styles: s.styles,
                lyrics: s.lyrics,
                duration: s.duration,
                createdAt: s.created_at * 1000, // Convert to ms
                albumArt: s.album_art, // Album art filename or null
            }));
            console.log(`Loaded ${songs.length} songs from server`);
        }
    } catch (e) {
        console.error('Failed to load songs:', e);
    }
}

// Get album art URL or gradient fallback
function getAlbumArtStyle(song) {
    if (song && song.albumArt) {
        return `background-image: url('${API_URL}/art/${song.albumArt}'); background-size: cover; background-position: center;`;
    }
    // Fallback gradient based on styles
    return getGradientFromStyles(song?.styles || '');
}

// Generate gradient from style tags
function getGradientFromStyles(styles) {
    const styleColors = {
        'electronic': ['#00d4ff', '#0066ff'],
        'ambient': ['#2d5a27', '#1a3a1a'],
        'rock': ['#ff4444', '#aa2222'],
        'pop': ['#ff69b4', '#ff1493'],
        'jazz': ['#9b59b6', '#8e44ad'],
        'classical': ['#f5e6d3', '#c9a86c'],
        'hip hop': ['#ffd700', '#ff8c00'],
        'rap': ['#ff6600', '#cc3300'],
        'metal': ['#333333', '#1a1a1a'],
        'acoustic': ['#8b4513', '#654321'],
        'piano': ['#2c3e50', '#1a252f'],
        'instrumental': ['#1a1a2e', '#16213e'],
    };

    const lowerStyles = styles.toLowerCase();
    for (const [style, colors] of Object.entries(styleColors)) {
        if (lowerStyles.includes(style)) {
            return `background: linear-gradient(135deg, ${colors[0]} 0%, ${colors[1]} 100%);`;
        }
    }
    // Default gradient
    return 'background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);';
}

// Check if generation is in progress (for page refresh recovery)
async function checkGenerationStatus() {
    try {
        const response = await fetch(`${API_URL}/progress`);
        if (response.ok) {
            const status = await response.json();
            if (status.is_generating) {
                isGenerating = true;
                showGenerationInProgress(status);
                startProgressPolling();
            }
        }
    } catch (e) {
        console.log('API not available');
    }
}

// Show generation in progress UI
function showGenerationInProgress(status) {
    const createBtn = document.getElementById('create-btn');
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    if (createBtn) {
        createBtn.classList.add('generating');
        createBtn.disabled = true;
        createBtn.innerHTML = '<svg class="spin" viewBox="0 0 24 24" fill="currentColor"><path d="M12 4V2A10 10 0 0 0 2 12h2a8 8 0 0 1 8-8z"/></svg> Generating...';
    }

    if (progressContainer) {
        progressContainer.classList.remove('hidden');
        progressFill.style.width = `${status.progress * 100}%`;
        progressText.textContent = status.message;

        // Show what's being generated
        if (status.current_request) {
            const req = status.current_request;
            const info = req.title || req.styles || 'Music';
            progressText.textContent = `${status.message} - "${info}"`;
        }
    }
}

// Start polling for progress
function startProgressPolling() {
    if (progressPollInterval) return;

    progressPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/progress`);
            if (response.ok) {
                const status = await response.json();

                const progressFill = document.getElementById('progress-fill');
                const progressText = document.getElementById('progress-text');

                if (progressFill) {
                    progressFill.style.width = `${status.progress * 100}%`;
                }
                if (progressText) {
                    let msg = status.message;
                    if (status.current_request) {
                        const req = status.current_request;
                        const info = req.title || req.styles.substring(0, 30) || 'Music';
                        msg = `${status.message}`;
                    }
                    progressText.textContent = msg;
                }

                if (!status.is_generating) {
                    stopProgressPolling();
                    isGenerating = false;
                    resetCreateButton();
                    // Reload songs after generation completes
                    await loadSongsFromServer();
                    renderWorkspaceList();
                }
            }
        } catch (e) {
            // Ignore polling errors
        }
    }, 500);
}

// Stop polling for progress
function stopProgressPolling() {
    if (progressPollInterval) {
        clearInterval(progressPollInterval);
        progressPollInterval = null;
    }
}

// Reset create button to normal state
function resetCreateButton() {
    const createBtn = document.getElementById('create-btn');
    if (createBtn) {
        createBtn.classList.remove('generating');
        createBtn.disabled = false;
        createBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7.5 5.6L10 7 8.6 4.5 10 2 7.5 3.4 5 2l1.4 2.5L5 7zm12 9.8L17 14l1.4 2.5L17 19l2.5-1.4L22 19l-1.4-2.5L22 14z"/></svg> Create';
    }

    const progressFill = document.getElementById('progress-fill');
    if (progressFill) {
        progressFill.style.width = '100%';
    }

    // Reset cancel button
    const cancelBtn = document.getElementById('cancel-btn');
    if (cancelBtn) {
        cancelBtn.disabled = false;
        cancelBtn.textContent = 'Cancel';
    }
}

// Art generation status polling
let artPollInterval = null;

function startArtStatusPolling() {
    if (artPollInterval) return;

    artPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/art-status`);
            if (response.ok) {
                const status = await response.json();
                updateArtStatusUI(status);
            }
        } catch (e) {
            // Ignore errors
        }
    }, 2000);

    // Immediate check
    checkArtStatus();
}

function stopArtStatusPolling() {
    if (artPollInterval) {
        clearInterval(artPollInterval);
        artPollInterval = null;
    }
}

async function checkArtStatus() {
    try {
        const response = await fetch(`${API_URL}/art-status`);
        if (response.ok) {
            const status = await response.json();
            updateArtStatusUI(status);
        }
    } catch (e) {
        // Ignore errors
    }
}

function updateArtStatusUI(status) {
    // Update the floating toast in bottom right
    const artToast = document.getElementById('art-toast');
    const artToastSubtitle = document.getElementById('art-toast-subtitle');

    if (!artToast) return;

    if (status.is_generating || status.pending_count > 0) {
        artToast.classList.remove('hidden');
        let subtitle = 'Processing...';
        if (status.current_song) {
            // Extract readable name from filename
            const name = status.current_song.replace(/^generation_/, '').replace(/\.wav$/, '');
            subtitle = name;
        }
        if (status.pending_count > 1) {
            subtitle += ` (+${status.pending_count - 1} more)`;
        }
        if (artToastSubtitle) artToastSubtitle.textContent = subtitle;
    } else {
        artToast.classList.add('hidden');
    }
}

// Initialize sliders
function initSliders() {
    const cfgSlider = document.getElementById('cfg-slider');
    const cfgValue = document.getElementById('cfg-value');
    const durationSlider = document.getElementById('duration-slider');
    const durationValue = document.getElementById('duration-value');
    const temperatureSlider = document.getElementById('temperature-slider');
    const temperatureValue = document.getElementById('temperature-value');

    if (cfgSlider && cfgValue) {
        cfgSlider.addEventListener('input', () => {
            cfgValue.textContent = cfgSlider.value;
        });
    }

    if (durationSlider && durationValue) {
        durationSlider.addEventListener('input', () => {
            durationValue.textContent = durationSlider.value + 's';
        });
    }

    if (temperatureSlider && temperatureValue) {
        temperatureSlider.addEventListener('input', () => {
            temperatureValue.textContent = temperatureSlider.value;
        });
    }
}

// Initialize mode toggle (Simple/Custom)
function initModeToggle() {
    const modeButtons = document.querySelectorAll('.mode-btn');
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.textContent.toLowerCase();
            setMode(mode);
        });
    });
}

// Set mode (simple/custom)
function setMode(mode) {
    currentMode = mode;
    const modeButtons = document.querySelectorAll('.mode-btn');
    modeButtons.forEach(btn => {
        btn.classList.toggle('active', btn.textContent.toLowerCase() === mode);
    });

    const advancedSection = document.querySelector('.section.collapsed, .section:has(#advanced-section)');
    const tagsRow = document.querySelector('.tags-row');

    if (mode === 'simple') {
        if (advancedSection) advancedSection.style.display = 'none';
        if (tagsRow) tagsRow.style.display = 'none';
        const durationSlider = document.getElementById('duration-slider');
        const durationValue = document.getElementById('duration-value');
        const cfgSlider = document.getElementById('cfg-slider');
        const cfgValue = document.getElementById('cfg-value');
        if (durationSlider) durationSlider.value = 30;
        if (durationValue) durationValue.textContent = '30s';
        if (cfgSlider) cfgSlider.value = 1.5;
        if (cfgValue) cfgValue.textContent = '1.5';
    } else {
        if (advancedSection) advancedSection.style.display = 'block';
        if (tagsRow) tagsRow.style.display = 'flex';
    }
}

// Initialize navigation
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const text = item.textContent.trim().toLowerCase();
            setView(text);
        });
    });
}

// Set current view
function setView(view) {
    currentView = view;

    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        const text = item.textContent.trim().toLowerCase();
        item.classList.toggle('active', text === view);
    });

    const mainContent = document.querySelector('.main-content');
    const workspace = document.querySelector('.workspace');

    if (view === 'create') {
        mainContent.innerHTML = getCreateViewHTML();
        workspace.style.display = 'block';
        initSliders();
        initModeToggle();
        setMode(currentMode);
        renderWorkspaceList();
        // Re-check generation status when switching to create view
        checkGenerationStatus();
    } else if (view === 'library') {
        mainContent.innerHTML = getLibraryViewHTML();
        workspace.style.display = 'none';
        renderLibraryView();
    } else if (view === 'search') {
        mainContent.innerHTML = getSearchViewHTML();
        workspace.style.display = 'none';
        initGlobalSearch();
    } else if (view === 'home') {
        mainContent.innerHTML = getHomeViewHTML();
        workspace.style.display = 'none';
    } else if (view === 'studio') {
        mainContent.innerHTML = getStudioViewHTML();
        workspace.style.display = 'none';
    } else if (view === 'settings') {
        mainContent.innerHTML = getSettingsViewHTML();
        workspace.style.display = 'none';
    }
}

// Get Create view HTML
function getCreateViewHTML() {
    return `
        <section class="section">
            <div class="section-header" onclick="toggleSection('lyrics')">
                <svg class="chevron" viewBox="0 0 24 24" fill="currentColor"><path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/></svg>
                <span>Lyrics</span>
            </div>
            <div class="section-content" id="lyrics-section">
                <textarea id="lyrics-input" placeholder="Write some lyrics or a prompt — or leave blank for instrumental" rows="6"></textarea>
            </div>
        </section>

        <section class="section">
            <div class="section-header" onclick="toggleSection('styles')">
                <svg class="chevron" viewBox="0 0 24 24" fill="currentColor"><path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/></svg>
                <span>Styles</span>
            </div>
            <div class="section-content" id="styles-section">
                <input type="text" id="styles-input" placeholder="pop, rock, electronic..." value="electronic, ambient, instrumental">
                <div class="tags-row">
                    <button class="tag-pill" onclick="addTag('pop')">+ pop</button>
                    <button class="tag-pill" onclick="addTag('rock')">+ rock</button>
                    <button class="tag-pill" onclick="addTag('jazz')">+ jazz</button>
                    <button class="tag-pill" onclick="addTag('piano')">+ piano</button>
                    <button class="tag-pill" onclick="addTag('acoustic')">+ acoustic</button>
                    <button class="tag-pill" onclick="addTag('female vocal')">+ female vocal</button>
                    <button class="tag-pill" onclick="addTag('male vocal')">+ male vocal</button>
                    <button class="tag-pill" onclick="addTag('emotional')">+ emotional</button>
                </div>
            </div>
        </section>

        <section class="section collapsed" style="${currentMode === 'simple' ? 'display:none' : ''}">
            <div class="section-header" onclick="toggleSection('advanced')">
                <svg class="chevron" viewBox="0 0 24 24" fill="currentColor"><path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/></svg>
                <span>Advanced Options</span>
            </div>
            <div class="section-content hidden" id="advanced-section">
                <div class="slider-row">
                    <span class="slider-label">Weirdness</span>
                    <div class="slider-container">
                        <input type="range" id="temperature-slider" min="0.5" max="1.5" step="0.1" value="1.0">
                        <span class="slider-value" id="temperature-value">1.0</span>
                    </div>
                </div>
                <div class="slider-row">
                    <span class="slider-label">Style Influence</span>
                    <div class="slider-container">
                        <input type="range" id="cfg-slider" min="1" max="4" step="0.1" value="1.5">
                        <span class="slider-value" id="cfg-value">1.5</span>
                    </div>
                </div>
                <div class="slider-row">
                    <span class="slider-label">Duration (seconds)</span>
                    <div class="slider-container">
                        <input type="range" id="duration-slider" min="5" max="120" step="5" value="30">
                        <span class="slider-value" id="duration-value">30s</span>
                    </div>
                </div>
                <div class="option-row">
                    <div class="option-icon">
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
                    </div>
                    <input type="text" placeholder="Song Title (Optional)" class="option-input" id="song-title">
                </div>
            </div>
        </section>

        <button class="create-btn" id="create-btn" onclick="generateMusic()" ${isGenerating ? 'disabled' : ''}>
            ${isGenerating ? '<svg class="spin" viewBox="0 0 24 24" fill="currentColor"><path d="M12 4V2A10 10 0 0 0 2 12h2a8 8 0 0 1 8-8z"/></svg> Generating...' : '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7.5 5.6L10 7 8.6 4.5 10 2 7.5 3.4 5 2l1.4 2.5L5 7zm12 9.8L17 14l1.4 2.5L17 19l2.5-1.4L22 19l-1.4-2.5L22 14z"/></svg> Create'}
        </button>

        <div class="progress-container ${isGenerating ? '' : 'hidden'}" id="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <span class="progress-text" id="progress-text">Generating...</span>
        </div>

        <div class="output-section hidden" id="output-section">
            <audio controls id="audio-player"></audio>
            <div class="output-info" id="output-info"></div>
        </div>
    `;
}

// Get Library view HTML
function getLibraryViewHTML() {
    return `
        <div class="view-header">
            <h1>Library</h1>
            <p class="view-subtitle">All generated songs from outputs folder</p>
        </div>
        <div class="library-search">
            <div class="search-box large">
                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
                <input type="text" placeholder="Search your library..." id="library-search-input">
            </div>
        </div>
        <div class="library-grid" id="library-grid"></div>
    `;
}

// Get Search view HTML
function getSearchViewHTML() {
    return `
        <div class="view-header">
            <h1>Search</h1>
            <p class="view-subtitle">Find songs by title, styles, or lyrics</p>
        </div>
        <div class="search-view">
            <div class="search-box large">
                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
                <input type="text" placeholder="Search by title, styles, or lyrics..." id="global-search-input" autofocus>
            </div>
            <div class="search-results" id="search-results">
                <p class="search-hint">Start typing to search your songs...</p>
            </div>
        </div>
    `;
}

// Get Home view HTML
function getHomeViewHTML() {
    return `
        <div class="view-header">
            <h1>Welcome to HeartMuLa</h1>
            <p class="view-subtitle">AI Music Generation powered by MLX</p>
        </div>
        <div class="home-stats">
            <div class="stat-card">
                <div class="stat-value">${songs.length}</div>
                <div class="stat-label">Songs Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${formatTotalDuration()}</div>
                <div class="stat-label">Total Duration</div>
            </div>
        </div>
        <div class="home-section">
            <h2>Recent Creations</h2>
            <div class="recent-songs" id="recent-songs">
                ${songs.length === 0 ? '<p class="empty-hint">No songs yet. Create your first one!</p>' : songs.slice(0, 5).map(s => `
                    <div class="song-item" onclick="playSong('${s.filename}', '${escapeHtml(s.title)}')">
                        <div class="song-thumb" style="${getAlbumArtStyle(s)}"><span class="duration">${formatDuration(s.duration)}</span></div>
                        <div class="song-info"><h4>${escapeHtml(s.title)}</h4><p>${escapeHtml(s.styles)}</p></div>
                    </div>
                `).join('')}
            </div>
        </div>
        <button class="create-btn" onclick="setView('create')">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>
            Start Creating
        </button>
    `;
}

// Get Studio view HTML
function getStudioViewHTML() {
    return `
        <div class="view-header">
            <h1>Studio</h1>
            <p class="view-subtitle">Your music generation workspace</p>
        </div>
        <div class="studio-stats">
            <p>Songs in outputs folder: <strong>${songs.length}</strong></p>
            <p>Location: <code>outputs/</code></p>
        </div>
        <button class="action-btn" onclick="loadSongsFromServer().then(() => { renderWorkspaceList(); alert('Reloaded ' + songs.length + ' songs'); })">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
            Refresh Library
        </button>
    `;
}

// Get Settings view HTML
function getSettingsViewHTML() {
    return `
        <div class="view-header">
            <h1>Settings</h1>
            <p class="view-subtitle">Configure HeartMuLa</p>
        </div>
        <div class="settings-container">
            <div class="settings-section">
                <h3>Album Art</h3>
                <div class="setting-item">
                    <label>Auto-generate Album Art</label>
                    <input type="checkbox" id="setting-auto-art" checked>
                </div>
            </div>

            <div class="settings-section">
                <h3>Text Generation (Surprise Me)</h3>
                <div class="setting-item">
                    <label>Provider</label>
                    <select id="setting-llm-provider">
                        <option value="none" selected>Built-in templates</option>
                        <option value="anthropic">Anthropic (Claude)</option>
                        <option value="openai">OpenAI (GPT)</option>
                    </select>
                </div>
                <div class="setting-item" id="api-key-row" style="display: none;">
                    <label>API Key</label>
                    <input type="password" id="setting-api-key" placeholder="sk-...">
                </div>
            </div>
        </div>
    `;
}

// Format total duration
function formatTotalDuration() {
    const totalSeconds = songs.reduce((sum, s) => sum + (s.duration || 0), 0);
    const mins = Math.floor(totalSeconds / 60);
    const secs = Math.floor(totalSeconds % 60);
    return `${mins}m ${secs}s`;
}

// Initialize workspace search
function initSearch() {
    const searchInput = document.querySelector('.workspace .search-box input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            searchQuery = e.target.value.toLowerCase();
            renderWorkspaceList();
        });
    }
}

// Initialize global search
function initGlobalSearch() {
    const searchInput = document.getElementById('global-search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            renderSearchResults(e.target.value.toLowerCase());
        });
    }
}

// Render search results
function renderSearchResults(query) {
    const resultsContainer = document.getElementById('search-results');
    if (!resultsContainer) return;

    if (!query) {
        resultsContainer.innerHTML = '<p class="search-hint">Start typing to search your songs...</p>';
        return;
    }

    const filtered = songs.filter(song =>
        song.title.toLowerCase().includes(query) ||
        song.styles.toLowerCase().includes(query) ||
        (song.lyrics && song.lyrics.toLowerCase().includes(query))
    );

    if (filtered.length === 0) {
        resultsContainer.innerHTML = `<p class="search-hint">No songs found for "${escapeHtml(query)}"</p>`;
        return;
    }

    resultsContainer.innerHTML = filtered.map(song => `
        <div class="search-result-item" onclick="playSong('${song.filename}', '${escapeHtml(song.title)}')">
            <div class="result-thumb" style="${getAlbumArtStyle(song)}"><span class="duration">${formatDuration(song.duration)}</span></div>
            <div class="result-info">
                <h4>${escapeHtml(song.title)}</h4>
                <p>${escapeHtml(song.styles)}</p>
            </div>
            <button class="play-btn-small"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg></button>
        </div>
    `).join('');
}

// Render library view
function renderLibraryView() {
    const grid = document.getElementById('library-grid');
    if (!grid) return;

    const librarySearch = document.getElementById('library-search-input');
    if (librarySearch) {
        librarySearch.addEventListener('input', (e) => {
            renderLibraryGrid(e.target.value.toLowerCase());
        });
    }

    renderLibraryGrid('');
}

// Render library grid
function renderLibraryGrid(filter) {
    const grid = document.getElementById('library-grid');
    if (!grid) return;

    const filtered = filter
        ? songs.filter(s => s.title.toLowerCase().includes(filter) || s.styles.toLowerCase().includes(filter))
        : songs;

    if (filtered.length === 0) {
        grid.innerHTML = `<p class="empty-hint">${filter ? 'No songs match your search' : 'No songs in outputs folder'}</p>`;
        return;
    }

    grid.innerHTML = filtered.map(song => `
        <div class="library-card" onclick="playSong('${song.filename}', '${escapeHtml(song.title)}')">
            <div class="library-card-thumb" style="${getAlbumArtStyle(song)}">
                <span class="duration">${formatDuration(song.duration)}</span>
                <button class="play-overlay"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg></button>
            </div>
            <div class="library-card-info">
                <h4>${escapeHtml(song.title)}</h4>
                <p>${escapeHtml(song.styles || 'No styles')}</p>
            </div>
        </div>
    `).join('');
}

// Initialize workspace controls
function initWorkspaceControls() {
    const sortBtn = document.querySelector('.sort-btn');
    if (sortBtn) {
        sortBtn.addEventListener('click', toggleSort);
    }
}

// Toggle sort order
function toggleSort() {
    const orders = ['newest', 'oldest', 'name'];
    const currentIndex = orders.indexOf(sortOrder);
    sortOrder = orders[(currentIndex + 1) % orders.length];

    const sortBtn = document.querySelector('.sort-btn');
    if (sortBtn) {
        sortBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 18h6v-2H3v2zM3 6v2h18V6H3zm0 7h12v-2H3v2z"/></svg>
            ${sortOrder.charAt(0).toUpperCase() + sortOrder.slice(1)}
        `;
    }

    renderWorkspaceList();
}

// Toggle section collapse
function toggleSection(section) {
    const content = document.getElementById(`${section}-section`);
    if (!content) return;
    const sectionEl = content.closest('.section');

    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        sectionEl.classList.remove('collapsed');
    } else {
        content.classList.add('hidden');
        sectionEl.classList.add('collapsed');
    }
}

// Add tag to styles input
function addTag(tag) {
    const input = document.getElementById('styles-input');
    if (!input) return;
    const currentTags = input.value.split(',').map(t => t.trim()).filter(t => t);

    if (!currentTags.includes(tag)) {
        currentTags.push(tag);
        input.value = currentTags.join(', ');
    }
}

// Set vocal gender - updates styles and toggle buttons
function setVocalGender(gender) {
    const input = document.getElementById('styles-input');
    if (!input) return;

    // Parse current tags
    let tags = input.value.split(',').map(t => t.trim()).filter(t => t);

    // Remove existing vocal gender tags
    tags = tags.filter(t => t !== 'male vocal' && t !== 'female vocal');

    // Add selected gender
    tags.push(`${gender} vocal`);
    input.value = tags.join(', ');

    // Update toggle buttons
    const buttons = document.querySelectorAll('.option-row .toggle-btn');
    buttons.forEach(btn => {
        const isMale = btn.textContent.trim().toLowerCase() === 'male';
        btn.classList.toggle('active', (gender === 'male') === isMale);
    });
}

// Surprise Me - fill in random lyrics and styles
async function surpriseMe() {
    try {
        const response = await fetch(`${API_URL}/surprise-lyrics`);
        if (response.ok) {
            const data = await response.json();

            // Fill in lyrics
            const lyricsInput = document.getElementById('lyrics-input');
            if (lyricsInput) {
                lyricsInput.value = data.lyrics;
            }

            // Fill in styles
            const stylesInput = document.getElementById('styles-input');
            if (stylesInput) {
                stylesInput.value = data.styles;
            }

            // Fill in title
            const titleInput = document.getElementById('title-input');
            if (titleInput) {
                titleInput.value = data.title;
            }

            // Visual feedback - flash the magic button
            const magicBtn = document.querySelector('.magic-btn');
            if (magicBtn) {
                magicBtn.style.transform = 'scale(1.2)';
                setTimeout(() => { magicBtn.style.transform = ''; }, 200);
            }
        }
    } catch (e) {
        console.error('Surprise failed:', e);
    }
}

// Cancel music generation
async function cancelGeneration() {
    try {
        const response = await fetch(`${API_URL}/cancel`, { method: 'POST' });
        if (response.ok) {
            const result = await response.json();
            console.log('Cancel result:', result);

            // Update UI immediately
            const progressText = document.getElementById('progress-text');
            if (progressText) progressText.textContent = 'Cancelling...';

            // Disable cancel button
            const cancelBtn = document.getElementById('cancel-btn');
            if (cancelBtn) {
                cancelBtn.disabled = true;
                cancelBtn.textContent = 'Cancelling...';
            }
        }
    } catch (e) {
        console.error('Failed to cancel:', e);
    }
}

// Generate music
async function generateMusic() {
    if (isGenerating) {
        alert('Generation already in progress. Please wait.');
        return;
    }

    const lyrics = document.getElementById('lyrics-input')?.value || '';
    const styles = document.getElementById('styles-input')?.value || 'electronic, ambient';
    const duration = parseInt(document.getElementById('duration-slider')?.value || '30');
    const cfgScale = parseFloat(document.getElementById('cfg-slider')?.value || '1.5');
    const temperature = parseFloat(document.getElementById('temperature-slider')?.value || '1.0');
    const title = document.getElementById('song-title')?.value || '';

    // Show progress immediately
    isGenerating = true;
    const createBtn = document.getElementById('create-btn');
    createBtn.classList.add('generating');
    createBtn.disabled = true;
    createBtn.innerHTML = '<svg class="spin" viewBox="0 0 24 24" fill="currentColor"><path d="M12 4V2A10 10 0 0 0 2 12h2a8 8 0 0 1 8-8z"/></svg> Generating...';

    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    progressContainer.classList.remove('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = 'Starting...';

    // Start polling
    startProgressPolling();

    try {
        const response = await fetch(`${API_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                lyrics,
                styles,
                duration,
                cfg_scale: cfgScale,
                temperature: temperature,
                topk: 50,
                title: title || `Generation ${songs.length + 1}`,
            })
        });

        if (response.status === 409) {
            alert('Generation already in progress on server');
            return;
        }

        if (!response.ok) {
            throw new Error('Generation failed');
        }

        const result = await response.json();

        // Reload songs from server
        await loadSongsFromServer();

        // Show output
        const outputSection = document.getElementById('output-section');
        const audioPlayer = document.getElementById('audio-player');
        const outputInfo = document.getElementById('output-info');

        if (audioPlayer && outputSection && outputInfo) {
            audioPlayer.src = `${API_URL}/audio/${result.filename}`;
            outputInfo.textContent = `${result.frames} frames (${result.duration.toFixed(1)}s) in ${result.time.toFixed(1)}s`;
            outputSection.classList.remove('hidden');
        }

        // Show player bar
        const playerBar = document.getElementById('player-bar');
        const playerTitle = document.getElementById('player-title');
        const playIcon = document.getElementById('play-icon');
        const playBtn = document.querySelector('.play-btn');
        if (playerBar && playerTitle) {
            playerBar.classList.remove('hidden');
            playerTitle.textContent = title || result.filename;
            document.body.style.paddingBottom = '90px';
            if (playIcon) playIcon.innerHTML = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>';
            if (playBtn) playBtn.classList.add('playing');
        }

        renderWorkspaceList();

    } catch (error) {
        console.error('Generation error:', error);
        progressText.textContent = `Error: ${error.message}`;
    } finally {
        stopProgressPolling();
        isGenerating = false;
        resetCreateButton();
    }
}

// Render workspace song list
function renderWorkspaceList() {
    const workspaceList = document.getElementById('workspace-list');
    if (!workspaceList) return;

    let displaySongs = [...songs];

    // Apply search filter
    if (searchQuery) {
        displaySongs = displaySongs.filter(s =>
            s.title.toLowerCase().includes(searchQuery) ||
            s.styles.toLowerCase().includes(searchQuery)
        );
    }

    // Apply sort
    if (sortOrder === 'oldest') {
        displaySongs.sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
    } else if (sortOrder === 'name') {
        displaySongs.sort((a, b) => a.title.localeCompare(b.title));
    } else {
        displaySongs.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
    }

    if (displaySongs.length === 0) {
        workspaceList.innerHTML = `
            <div class="empty-state">
                <p>${searchQuery ? 'No songs match your search' : 'No songs yet. Create your first one!'}</p>
            </div>
        `;
        return;
    }

    workspaceList.innerHTML = displaySongs.map(song => `
        <div class="song-item">
            <div class="song-thumb" style="${getAlbumArtStyle(song)}"><span class="duration">${formatDuration(song.duration)}</span></div>
            <div class="song-info">
                <h4>${escapeHtml(song.title)}</h4>
                <p>${escapeHtml(song.styles || 'No styles')}</p>
            </div>
            <div class="song-actions">
                <button class="song-action-btn" onclick="playSong('${song.filename}', '${escapeHtml(song.title)}')">
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

// Play song
function playSong(filename, title) {
    const audioPlayer = document.getElementById('audio-player');
    const playerBar = document.getElementById('player-bar');
    const playerTitle = document.getElementById('player-title');
    const playIcon = document.getElementById('play-icon');
    const playBtn = document.querySelector('.play-btn');
    const trackThumb = document.getElementById('track-thumb');

    // Find current track and index
    currentTrackIndex = songs.findIndex(s => s.filename === filename);
    const currentSong = songs[currentTrackIndex];

    if (!audioPlayer) {
        const audio = new Audio(`${API_URL}/audio/${filename}`);
        audio.play();
        return;
    }

    audioPlayer.src = `${API_URL}/audio/${filename}`;
    audioPlayer.play();

    // Update play button to pause icon
    if (playIcon) playIcon.innerHTML = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>';
    if (playBtn) playBtn.classList.add('playing');

    // Update track thumbnail with album art
    if (trackThumb && currentSong) {
        trackThumb.style.cssText = getAlbumArtStyle(currentSong);
    }

    if (playerBar) playerBar.classList.remove('hidden');
    if (playerTitle) playerTitle.textContent = title;
    document.body.style.paddingBottom = '90px';
}

// Player state
let currentTrackIndex = -1;
let isShuffled = false;
let repeatMode = 'none'; // 'none', 'all', 'one'
let isMuted = false;
let previousVolume = 80;

// Toggle play/pause
function togglePlay() {
    const audioPlayer = document.getElementById('audio-player');
    const playIcon = document.getElementById('play-icon');
    const playBtn = document.querySelector('.play-btn');

    if (!audioPlayer || !audioPlayer.src) return;

    if (audioPlayer.paused) {
        audioPlayer.play();
        if (playIcon) playIcon.innerHTML = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>';
        if (playBtn) playBtn.classList.add('playing');
    } else {
        audioPlayer.pause();
        if (playIcon) playIcon.innerHTML = '<path d="M8 5v14l11-7z"/>';
        if (playBtn) playBtn.classList.remove('playing');
    }
}

// Update progress bar and time display
function updateProgress() {
    const audioPlayer = document.getElementById('audio-player');
    const progressPlayed = document.getElementById('progress-played');
    const progressHandle = document.getElementById('progress-handle');
    const timeCurrent = document.getElementById('time-current');
    const timeTotal = document.getElementById('time-total');

    if (!audioPlayer || !audioPlayer.duration) return;

    const percent = (audioPlayer.currentTime / audioPlayer.duration) * 100;
    if (progressPlayed) progressPlayed.style.width = `${percent}%`;
    if (timeCurrent) timeCurrent.textContent = formatDuration(audioPlayer.currentTime);
    if (timeTotal) timeTotal.textContent = formatDuration(audioPlayer.duration);
}

// Seek to position
function seekTo(event) {
    const audioPlayer = document.getElementById('audio-player');
    const progressTrack = document.getElementById('progress-track');

    if (!audioPlayer || !audioPlayer.duration || !progressTrack) return;

    const rect = progressTrack.getBoundingClientRect();
    const percent = (event.clientX - rect.left) / rect.width;
    audioPlayer.currentTime = percent * audioPlayer.duration;
}

// Set volume
function setVolume(value) {
    const audioPlayer = document.getElementById('audio-player');
    const volumeSlider = document.getElementById('volume-slider');
    const volumeIcon = document.getElementById('volume-icon');

    if (audioPlayer) {
        audioPlayer.volume = value / 100;
        isMuted = value == 0;
    }

    // Update slider visual
    if (volumeSlider) {
        volumeSlider.style.setProperty('--volume-percent', `${value}%`);
    }

    // Update icon
    if (volumeIcon) {
        if (value == 0) {
            volumeIcon.innerHTML = '<path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>';
        } else if (value < 50) {
            volumeIcon.innerHTML = '<path d="M18.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM5 9v6h4l5 5V4L9 9H5z"/>';
        } else {
            volumeIcon.innerHTML = '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>';
        }
    }
}

// Toggle mute
function toggleMute() {
    const audioPlayer = document.getElementById('audio-player');
    const volumeSlider = document.getElementById('volume-slider');

    if (!audioPlayer) return;

    if (isMuted || audioPlayer.volume === 0) {
        setVolume(previousVolume);
        if (volumeSlider) volumeSlider.value = previousVolume;
    } else {
        previousVolume = audioPlayer.volume * 100;
        setVolume(0);
        if (volumeSlider) volumeSlider.value = 0;
    }
}

// Toggle shuffle
function toggleShuffle() {
    isShuffled = !isShuffled;
    const btn = document.getElementById('shuffle-btn');
    if (btn) btn.classList.toggle('active', isShuffled);
}

// Toggle repeat
function toggleRepeat() {
    const modes = ['none', 'all', 'one'];
    const currentIndex = modes.indexOf(repeatMode);
    repeatMode = modes[(currentIndex + 1) % modes.length];

    const btn = document.getElementById('repeat-btn');
    if (btn) {
        btn.classList.toggle('active', repeatMode !== 'none');
        // Change icon for repeat one
        if (repeatMode === 'one') {
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4zm-4-2V9h-1l-2 1v1h1.5v4H13z"/></svg>';
        } else {
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z"/></svg>';
        }
    }
}

// Previous track
function previousTrack() {
    const audioPlayer = document.getElementById('audio-player');
    if (audioPlayer && audioPlayer.currentTime > 3) {
        audioPlayer.currentTime = 0;
        return;
    }
    if (currentTrackIndex > 0) {
        const prevSong = songs[currentTrackIndex - 1];
        playSong(prevSong.filename, prevSong.title);
        currentTrackIndex--;
    }
}

// Next track
function nextTrack() {
    if (currentTrackIndex < songs.length - 1) {
        const nextSong = songs[currentTrackIndex + 1];
        playSong(nextSong.filename, nextSong.title);
        currentTrackIndex++;
    } else if (repeatMode === 'all' && songs.length > 0) {
        const firstSong = songs[0];
        playSong(firstSong.filename, firstSong.title);
        currentTrackIndex = 0;
    }
}

// Toggle like
function toggleLike() {
    const btn = document.getElementById('like-btn');
    const dislikeBtn = document.getElementById('dislike-btn');
    if (btn) {
        btn.classList.toggle('active');
        if (btn.classList.contains('active') && dislikeBtn) {
            dislikeBtn.classList.remove('active');
        }
    }
}

// Toggle dislike
function toggleDislike() {
    const btn = document.getElementById('dislike-btn');
    const likeBtn = document.getElementById('like-btn');
    if (btn) {
        btn.classList.toggle('active');
        if (btn.classList.contains('active') && likeBtn) {
            likeBtn.classList.remove('active');
        }
    }
}

// Download current track
function downloadTrack() {
    const audioPlayer = document.getElementById('audio-player');
    if (audioPlayer && audioPlayer.src) {
        const a = document.createElement('a');
        a.href = audioPlayer.src;
        a.download = document.getElementById('player-title')?.textContent || 'track.wav';
        a.click();
    }
}

// Toggle queue panel (placeholder)
function toggleQueue() {
    console.log('Queue panel not yet implemented');
}

// Initialize audio player events
function initAudioPlayer() {
    const audioPlayer = document.getElementById('audio-player');
    if (!audioPlayer) return;

    audioPlayer.addEventListener('timeupdate', updateProgress);
    audioPlayer.addEventListener('loadedmetadata', updateProgress);
    audioPlayer.addEventListener('ended', () => {
        const playIcon = document.getElementById('play-icon');
        const playBtn = document.querySelector('.play-btn');
        if (playIcon) playIcon.innerHTML = '<path d="M8 5v14l11-7z"/>';
        if (playBtn) playBtn.classList.remove('playing');

        if (repeatMode === 'one') {
            audioPlayer.currentTime = 0;
            audioPlayer.play();
        } else {
            nextTrack();
        }
    });

    // Set initial volume
    setVolume(80);
}

// Format duration
function formatDuration(seconds) {
    if (!seconds || isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Escape HTML
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add spin animation
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .spin { animation: spin 1s linear infinite; }
    .studio-stats { margin-bottom: 24px; }
    .studio-stats p { margin-bottom: 8px; color: var(--text-secondary); }
    .studio-stats code { background: var(--bg-tertiary); padding: 2px 8px; border-radius: 4px; }
`;
document.head.appendChild(style);
