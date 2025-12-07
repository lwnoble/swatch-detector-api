"""
Flask app for swatch detection with CORS support
Deployed on Render with Docker
"""

import sys
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image

print("Starting imports...", file=sys.stderr)

try:
    print("✓ Flask imported", file=sys.stderr)
except Exception as e:
    print(f"✗ Flask import failed: {e}", file=sys.stderr)
    traceback.print_exc()

try:
    import cv2
    print("✓ cv2 imported", file=sys.stderr)
except Exception as e:
    print(f"✗ cv2 import failed: {e}", file=sys.stderr)
    traceback.print_exc()

print("Creating Flask app...", file=sys.stderr)
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    },
    r"/health": {
        "origins": "*",
        "methods": ["GET", "OPTIONS"]
    }
})

print("Flask app created successfully with CORS enabled", file=sys.stderr)


class SwatchDetector:
    """Detects color swatches in images"""

    def __init__(self, image_array, min_swatch_area=5000):
        self.image = image_array
        self.height, self.width = self.image.shape[:2]
        self.min_swatch_area = min_swatch_area
        self.swatches = []

    def detect_swatches(self):
        """Main detection pipeline"""
        blurred = cv2.bilateralFilter(self.image, 9, 75, 75)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_swatch_area or area > self.width * self.height * 0.5:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            shape_type = self._classify_shape(contour, w, h)

            if shape_type:
                mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_color = cv2.mean(self.image, mask=mask)[:3]
                rgb = tuple(int(c) for c in reversed(mean_color))
                hex_color = self._rgb_to_hex(rgb)

                self.swatches.append({
                    'type': shape_type,
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': int(area),
                    'color_rgb': rgb,
                    'color_hex': hex_color
                })

        self.swatches.sort(key=lambda s: s['area'], reverse=True)
        return self.swatches

    def _classify_shape(self, contour, w, h):
        """Classify shape as circle, square, or rectangle"""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = float(w) / h if h > 0 else 0

        if circularity > 0.75:
            return 'circle'
        if num_vertices == 4:
            return 'square' if 0.85 < aspect_ratio < 1.15 else 'rectangle'
        if 3 <= num_vertices <= 5 and 0.5 < aspect_ratio < 2.0:
            return 'rectangle'
        return None

    def _rgb_to_hex(self, rgb):
        """Convert RGB to hex"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'swatch-detector'})


@app.route('/api/detect-swatches', methods=['POST', 'OPTIONS'])
def detect_swatches():
    """Detect swatches in an image"""
    
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image_data = data['image']
        min_swatch_area = data.get('min_swatch_area', 5000)

        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_data = image_data.split(',')[1]

        try:
            image_bytes = base64.b64decode(image_data)
            image_pil = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid image format: {str(e)}'}), 400

        image_array = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        detector = SwatchDetector(image_array, min_swatch_area=min_swatch_area)
        swatches = detector.detect_swatches()

        response_swatches = [
            {
                'type': s['type'],
                'x': s['x'],
                'y': s['y'],
                'width': s['width'],
                'height': s['height'],
                'color_hex': s['color_hex'],
                'color_rgb': s['color_rgb'],
                'area': s['area']
            }
            for s in swatches
        ]

        return jsonify({
            'success': True,
            'swatches': response_swatches,
            'count': len(response_swatches)
        }), 200

    except Exception as e:
        print(f"Error in detect_swatches: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("Running Flask app on 0.0.0.0:8000", file=sys.stderr)
    app.run(host='0.0.0.0', port=8000, debug=False)
