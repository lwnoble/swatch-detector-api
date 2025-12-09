<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 600px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 28px;
            margin-bottom: 8px;
        }

        .header p {
            opacity: 0.9;
            font-size: 14px;
        }

        .drop-zone {
            background: white;
            border: 3px dashed #667eea;
            border-radius: 12px;
            padding: 50px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .drop-zone:hover {
            border-color: #764ba2;
            background: rgba(102, 126, 234, 0.05);
        }

        .drop-zone.active {
            border-color: #764ba2;
            background: rgba(102, 126, 234, 0.1);
            transform: scale(1.02);
        }

        .drop-zone-content {
            pointer-events: none;
        }

        .drop-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }

        .drop-text {
            color: #333;
            margin-bottom: 8px;
        }

        .drop-subtext {
            color: #999;
            font-size: 13px;
        }

        #fileInput {
            display: none;
        }

        .preview-section {
            margin-top: 30px;
            display: none;
        }

        .preview-section.active {
            display: block;
        }

        .preview-image {
            width: 100%;
            max-height: 300px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: white;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .swatches-container {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-top: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }

        .swatches-header {
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid #eee;
        }

        .stat {
            text-align: center;
        }

        .stat-number {
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
        }

        .stat-label {
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .swatches-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }

        .swatch {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            position: relative;
        }

        .swatch-wrapper {
            position: relative;
            width: 80px;
        }

        .swatch-color {
            width: 80px;
            height: 80px;
            border-radius: 8px;
            border: 2px solid #eee;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .swatch:hover .swatch-color {
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            transform: scale(1.05);
        }

        .swatch-remove {
            position: absolute;
            top: -8px;
            right: -8px;
            width: 24px;
            height: 24px;
            background: #ff4444;
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.2s ease;
            font-weight: bold;
        }

        .swatch:hover .swatch-remove {
            opacity: 1;
        }

        .swatch-remove:hover {
            background: #dd0000;
        }

        .swatch-label {
            font-size: 11px;
            color: #666;
            text-align: center;
            word-break: break-all;
            font-weight: 500;
            width: 100%;
        }

        .swatch-type {
            font-size: 10px;
            color: #999;
            text-transform: capitalize;
        }

        .add-color-section {
            margin-bottom: 20px;
            padding: 16px;
            background: #f8f8f8;
            border-radius: 8px;
        }

        .add-color-title {
            font-size: 12px;
            font-weight: 600;
            color: #333;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .add-color-inputs {
            display: flex;
            gap: 8px;
        }

        .color-picker-btn {
            flex: 1;
            padding: 10px;
            border: 2px solid #667eea;
            background: white;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            color: #667eea;
            transition: all 0.2s ease;
        }

        .color-picker-btn:hover {
            background: #667eea;
            color: white;
        }

        .hex-input {
            flex: 1;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 12px;
            font-family: monospace;
            transition: border-color 0.2s ease;
        }

        .hex-input:focus {
            outline: none;
            border-color: #667eea;
        }

        .hex-input::placeholder {
            color: #999;
        }

        #colorPickerHidden {
            display: none;
        }

        .actions {
            display: flex;
            gap: 12px;
            margin-top: 20px;
        }

        .btn {
            flex: 1;
            padding: 12px 16px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }

        .btn-secondary:hover {
            background: #e0e0e0;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .error {
            background: #fee;
            color: #c33;
            padding: 16px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
            font-size: 14px;
        }

        .error.active {
            display: block;
        }

        .success {
            background: #efe;
            color: #3c3;
            padding: 16px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
            font-size: 14px;
        }

        .success.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Swatch Detector</h1>
            <p>Upload a mood board to extract colors</p>
        </div>

        <div class="drop-zone" id="dropZone">
            <div class="drop-zone-content">
                <div class="drop-icon">📸</div>
                <div class="drop-text">Drop your image here</div>
                <div class="drop-subtext">or click to select a file</div>
            </div>
        </div>

        <input type="file" id="fileInput" accept="image/*">
        <input type="color" id="colorPickerHidden">

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Analyzing image...</div>
        </div>

        <div class="preview-section" id="previewSection">
            <img id="previewImage" class="preview-image" alt="Preview">

            <div class="swatches-container">
                <div class="stats" id="stats"></div>

                <div class="swatches-header">
                    Extracted Swatches
                </div>

                <div class="add-color-section">
                    <div class="add-color-title">Add Color</div>
                    <div class="add-color-inputs">
                        <button class="color-picker-btn" id="colorPickerBtn">+ Pick Color</button>
                        <input type="text" class="hex-input" id="hexInput" placeholder="#FF6B6B">
                        <button class="color-picker-btn" id="addHexBtn" style="flex: 0.8;">Add</button>
                    </div>
                </div>

                <div class="swatches-grid" id="swatchesGrid"></div>

                <div class="actions">
                    <button class="btn btn-primary" id="applyButton">Apply to Figma</button>
                    <button class="btn btn-secondary" id="resetButton">Upload New</button>
                </div>
            </div>
        </div>

        <div class="error" id="error"></div>
        <div class="success" id="success"></div>
    </div>

    <script>
        const API_URL = 'https://swatch-detector-api-11.onrender.com/api/detect-swatches';

        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const previewSection = document.getElementById('previewSection');
        const previewImage = document.getElementById('previewImage');
        const swatchesGrid = document.getElementById('swatchesGrid');
        const applyButton = document.getElementById('applyButton');
        const resetButton = document.getElementById('resetButton');
        const errorDiv = document.getElementById('error');
        const successDiv = document.getElementById('success');
        const statsDiv = document.getElementById('stats');
        const colorPickerBtn = document.getElementById('colorPickerBtn');
        const colorPickerHidden = document.getElementById('colorPickerHidden');
        const hexInput = document.getElementById('hexInput');
        const addHexBtn = document.getElementById('addHexBtn');

        let currentSwatches = [];

        // Drag and drop
        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('active');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('active');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('active');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        // Color picker button
        colorPickerBtn.addEventListener('click', () => {
            colorPickerHidden.click();
        });

        colorPickerHidden.addEventListener('change', (e) => {
            const color = e.target.value;
            addColorToSwatches(color);
        });

        // Hex input
        addHexBtn.addEventListener('click', () => {
            const hex = hexInput.value.trim();
            if (hex.match(/^#[0-9A-Fa-f]{6}$/)) {
                addColorToSwatches(hex);
                hexInput.value = '';
            } else {
                showError('Invalid hex code. Use format: #RRGGBB');
            }
        });

        hexInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                addHexBtn.click();
            }
        });

        function addColorToSwatches(hexColor) {
            const rgb = hexToRgb(hexColor);
            const newSwatch = {
                type: 'square',
                x: 0,
                y: 0,
                width: 80,
                height: 80,
                area: 6400,
                color_rgb: rgb,
                color_hex: hexColor.toUpperCase()
            };
            
            currentSwatches.unshift(newSwatch);
            displaySwatches(currentSwatches);
            showSuccess(`Added ${hexColor}`);
        }

        function hexToRgb(hex) {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? [
                parseInt(result[1], 16),
                parseInt(result[2], 16),
                parseInt(result[3], 16)
            ] : [0, 0, 0];
        }

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Please upload an image file');
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                detectSwatches(e.target.result);
            };
            reader.readAsDataURL(file);
        }

        async function detectSwatches(base64Image) {
            loading.classList.add('active');
            errorDiv.classList.remove('active');
            successDiv.classList.remove('active');

            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image: base64Image,
                        min_swatch_area: 400
                    })
                });

                const data = await response.json();

                if (!data.success) {
                    showError(data.error || 'Failed to detect swatches');
                    loading.classList.remove('active');
                    return;
                }

                currentSwatches = data.swatches;
                displaySwatches(data.swatches);
                previewSection.classList.add('active');
                dropZone.style.display = 'none';
                loading.classList.remove('active');

            } catch (error) {
                showError(`Error: ${error.message}`);
                loading.classList.remove('active');
            }
        }

        function displaySwatches(swatches) {
            swatchesGrid.innerHTML = '';

            // Count by type
            const typeCount = {};
            swatches.forEach(s => {
                typeCount[s.type] = (typeCount[s.type] || 0) + 1;
            });

            // Display stats
            statsDiv.innerHTML = `
                <div class="stat">
                    <div class="stat-number">${swatches.length}</div>
                    <div class="stat-label">Colors</div>
                </div>
                ${Object.entries(typeCount).map(([type, count]) => `
                    <div class="stat">
                        <div class="stat-number">${count}</div>
                        <div class="stat-label">${type}s</div>
                    </div>
                `).join('')}
            `;

            // Display swatches
            swatches.forEach((swatch, index) => {
                const swatchEl = document.createElement('div');
                swatchEl.className = 'swatch';
                swatchEl.innerHTML = `
                    <div class="swatch-wrapper">
                        <div class="swatch-color" style="background-color: ${swatch.color_hex};" title="${swatch.color_hex}"></div>
                        <button class="swatch-remove" data-index="${index}">×</button>
                    </div>
                    <div class="swatch-label">${swatch.color_hex}</div>
                    <div class="swatch-type">${swatch.type}</div>
                `;
                
                swatchEl.querySelector('.swatch-color').addEventListener('click', () => copyToClipboard(swatch.color_hex));
                swatchEl.querySelector('.swatch-remove').addEventListener('click', () => removeSwatch(index));
                
                swatchesGrid.appendChild(swatchEl);
            });
        }

        function removeSwatch(index) {
            currentSwatches.splice(index, 1);
            displaySwatches(currentSwatches);
            showSuccess('Color removed');
        }

        function copyToClipboard(text) {
            navigator.clipboard.writeText(text);
            showSuccess(`Copied ${text}`);
        }

        applyButton.addEventListener('click', () => {
            if (currentSwatches.length === 0) {
                showError('No swatches to apply');
                return;
            }

            // Send swatches back to Figma plugin
            parent.postMessage({
                pluginMessage: {
                    type: 'apply-swatches',
                    swatches: currentSwatches
                }
            }, '*');

            showSuccess('Applied swatches to Figma!');
        });

        resetButton.addEventListener('click', () => {
            fileInput.value = '';
            previewSection.classList.remove('active');
            dropZone.style.display = 'block';
            currentSwatches = [];
            errorDiv.classList.remove('active');
            successDiv.classList.remove('active');
        });

        function showError(message) {
            errorDiv.textContent = message;
            errorDiv.classList.add('active');
            setTimeout(() => {
                errorDiv.classList.remove('active');
            }, 5000);
        }

        function showSuccess(message) {
            successDiv.textContent = message;
            successDiv.classList.add('active');
            setTimeout(() => {
                successDiv.classList.remove('active');
            }, 3000);
        }
    </script>
</body>
</html>
