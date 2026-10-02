/**
 * CattleAI Pro v2.0
 * Neural Inference & Recognition Controller
 * Frontend Logic
 */

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImage = document.getElementById('preview-image');
const scanLine = document.getElementById('scan-line');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingSection = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const mismatchSection = document.getElementById('mismatch-section');

// State Management
let selectedFile = null;
let currentPredictionId = null;

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadHistory();
    applyInitialAnimations();
});

function applyInitialAnimations() {
    // Reveal main container elements with a slight delay
    document.querySelectorAll('.glass').forEach((el, index) => {
        el.style.opacity = '0';
        setTimeout(() => {
            el.classList.add('fade-up');
            el.style.opacity = '1';
        }, index * 150);
    });
}

// Event Listeners
function setupEventListeners() {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    analyzeBtn.addEventListener('click', analyzImage);
}

// File Interaction Handlers
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) processFile(file);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showAlert('Protocol Error: High-resolution image required.', 'error');
    }
}

// core Processing
function processFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showAlert('Invalid Data Format. Please use JPG, PNG, WEBP, or BMP.', 'error');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showAlert('Payload Excessive. Maximum allowed size is 10MB.', 'error');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);

        previewSection.style.display = 'block';
        previewSection.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // Reset analysis UI
        resultsSection.style.display = 'none';
        mismatchSection.style.display = 'none';
        scanLine.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' bytes';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Neural Analysis
async function analyzImage() {
    if (!selectedFile) {
        showAlert('Input Missing: Please select an asset first.', 'error');
        return;
    }

    // Begin Animation sequence
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = 'Analyzing...';

    // Show scan effect on preview
    scanLine.style.display = 'block';

    // Smooth transition to loading
    loadingSection.style.display = 'block';
    loadingSection.classList.add('fade-up');

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('save_to_db', 'true');

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            // Handle animal mismatch specifically
            if (response.status === 422 && error.detail && error.detail.error === 'animal_mismatch') {
                showAnimalMismatch(error.detail.detected_animal);
                return;
            }
            throw new Error(error.detail || 'Neural inference failed');
        }

        const result = await response.json();

        // Slight delay for effect
        setTimeout(() => {
            displayResults(result);
            currentPredictionId = result.prediction_id;
            loadHistory();
            scanLine.style.display = 'none';
        }, 1200);

    } catch (error) {
        console.error('Core Error:', error);
        showAlert('Neural Core Malfunction: ' + error.message, 'error');
        scanLine.style.display = 'none';
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '⚡ Initiate Neural Analysis';
        loadingSection.style.display = 'none';
    }
}

// UI Data Rendering
function displayResults(result) {
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-up');

    // Classification
    document.getElementById('animal-type-value').textContent = result.animal_type.prediction;
    updateConfidenceBar('animal-type', result.animal_type.confidence);

    // Breed Identification
    document.getElementById('breed-value').textContent = result.breed.prediction;
    updateConfidenceBar('breed', result.breed.confidence);

    const topBreedsHtml = result.breed.top_predictions.map(p => `
        <div class="prediction-item" style="display:flex; justify-content:space-between; margin-bottom:0.5rem; font-size:0.9rem;">
            <span style="font-weight:500;">${p.breed}</span>
            <span style="color:var(--primary-dark); font-weight:700;">${(p.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
    document.getElementById('top-breeds').innerHTML = topBreedsHtml;

    // Maturity Level
    document.getElementById('age-value').textContent = result.age.prediction;
    updateConfidenceBar('age', result.age.confidence);

    const agePredHtml = result.age.all_predictions.slice(0, 3).map(p => `
        <div class="prediction-item" style="display:flex; justify-content:space-between; margin-bottom:0.5rem; font-size:0.9rem;">
            <span style="font-weight:500;">${p.age_group}</span>
            <span style="color:var(--accent); font-weight:700;">${(p.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
    document.getElementById('age-predictions').innerHTML = agePredHtml;

    // Detailed Profile
    if (result.breed_info) {
        displayBreedInfo(result.breed.prediction, result.breed_info);
    }

    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateConfidenceBar(type, confidence) {
    const fill = document.getElementById(`${type}-confidence-fill`);
    const text = document.getElementById(`${type}-confidence-text`);
    const percentage = confidence * 100;

    // Trigger animation with a small timeout for smoothness
    setTimeout(() => {
        fill.style.width = percentage + '%';
        text.textContent = 'Confidence Rating: ' + percentage.toFixed(1) + '%';

        fill.className = 'confidence-fill';
        if (percentage >= 70) fill.classList.add('high');
        else if (percentage >= 40) fill.classList.add('medium');
        else fill.classList.add('low');
    }, 100);
}

function displayBreedInfo(breedName, info) {
    const section = document.getElementById('breed-info-section');
    section.style.display = 'block';
    document.getElementById('breed-info-name').textContent = breedName;

    const detailsHtml = `
        <div class="breed-detail-item glass">
            <label>Geographic Origin</label>
            <span>${info.origin || 'N/A'}</span>
        </div>
        <div class="breed-detail-item glass">
            <label>Economic Category</label>
            <span>${info.type || 'N/A'}</span>
        </div>
        <div class="breed-detail-item glass">
            <label>Productivity (Milk)</label>
            <span>${info.average_milk_yield || 'N/A'}</span>
        </div>
        <div class="breed-detail-item glass">
            <label>Key Features</label>
            <span>${info.characteristics || 'N/A'}</span>
        </div>
    `;
    document.getElementById('breed-details').innerHTML = detailsHtml;
}

// Feedback Interaction
async function submitFeedback(isCorrect) {
    if (!currentPredictionId) {
        showAlert('Context Error: Prediction history missing.', 'error');
        return;
    }

    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prediction_id: currentPredictionId,
                feedback: isCorrect ? 'correct' : 'incorrect'
            })
        });

        if (response.ok) {
            showAlert('Telemetry Received. Accuracy logged.', 'success');
        }
    } catch (error) {
        console.error('Feedback Error:', error);
    }
}

// history Management
async function loadHistory() {
    const historyBody = document.getElementById('history-body');
    if (!historyBody) return;

    try {
        const response = await fetch('/api/predictions?limit=10');
        const predictions = await response.json();

        if (predictions.length === 0) {
            historyBody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding:3rem; color:var(--text-muted);">Neural Logs Empty. Start analyzing assets.</td></tr>`;
            return;
        }

        historyBody.innerHTML = predictions.map(p => `
            <tr>
                <td><img src="${p.image_path}" class="history-image" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🐄</text></svg>'"></td>
                <td style="font-weight:700;">${p.animal_type}</td>
                <td>
                    <div style="font-weight:800; color:var(--primary-dark);">${p.predicted_breed}</div>
                    <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">${(p.breed_confidence * 100).toFixed(1)}% Conf.</div>
                </td>
                <td style="font-weight:600;">${p.predicted_age}</td>
                <td style="font-size:0.8rem; color:var(--text-muted); font-weight:500;">${formatDate(p.created_at)}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Sync Error:', error);
    }
}

async function clearHistory() {
    const ok = confirm('Clear all System Logs? This will delete all saved predictions.');
    if (!ok) return;

    try {
        const response = await fetch('/api/predictions', { method: 'DELETE' });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to clear logs');
        }
        currentPredictionId = null;
        await loadHistory();
        showAlert('System Logs cleared.', 'success');
    } catch (error) {
        console.error('Clear Logs Error:', error);
        showAlert('Failed to clear logs: ' + error.message, 'error');
    }
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' | ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Animal Mismatch Handler
function showAnimalMismatch(detectedAnimal) {
    resultsSection.style.display = 'none';
    mismatchSection.style.display = 'block';
    const detectedText = document.getElementById('mismatch-detected');
    if (detectedAnimal && detectedAnimal !== 'unknown animal') {
        detectedText.textContent = `Detected: "${detectedAnimal}" — Please upload a cow or buffalo image.`;
    } else {
        detectedText.textContent = 'Please upload a cow or buffalo image.';
    }
    mismatchSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function resetMismatch() {
    mismatchSection.style.display = 'none';
    fileInput.value = '';
    selectedFile = null;
    previewSection.style.display = 'none';
}

// System Alerts
function showAlert(message, type = 'info') {
    const alert = document.createElement('div');
    alert.className = `glass alert alert-${type} fade-up`;
    alert.style.position = 'fixed';
    alert.style.top = '2rem';
    alert.style.right = '2rem';
    alert.style.zIndex = '5000';
    alert.style.padding = '1rem 2rem';
    alert.style.borderLeft = `6px solid var(--${type === 'error' ? 'error' : type === 'success' ? 'success' : 'info'})`;

    alert.innerHTML = `<span style="font-weight:700;">${message}</span>`;
    document.body.appendChild(alert);
    setTimeout(() => alert.remove(), 4000);
}
