from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)


class SwatchDetector:
    def __init__(self, image_array, min_swatch_area=1500):
        self.image = image_array
        self.height, self.width = self.image.shape[:2]
        self.min_swatch_area = min_swatch_area
        self.swatches = []

    def detect_swatches(self):
        """Detect color swatches by finding distinct color regions"""
        # Get distinct colors in image
        swatches = self._find_color_regions()
        
        # Remove duplicates and noise
        swatches = self._remove_duplicates(swatches)
        swatches = self._remove_noise(swatches)
        
        self.swatches = swatches
        self.swatches.sort(key=lambda s: s['area'], reverse=True)
        
        return self.swatches

    def _find_color_regions(self):
        """Find contiguous regions of similar color"""
        # Reduce to fewer colors using K-means
        num_colors = 20
        pixels = self.image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        reduced = centers[labels.flatten()].reshape(self.image.shape)
        
        # Find contours in reduced image
        gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding for better edge detection
        edges = cv2.Canny(gray, 20, 80)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        swatches = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum size
            if area < self.min_swatch_area:
                continue
            
            # Skip very large regions (likely background/photos)
            if area > self.width * self.height * 0.6:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get average color of this region
            region = self.image[y:y+h, x:x+w]
            avg_color = np.mean(region, axis=(0, 1))
            rgb = tuple(int(c) for c in reversed(avg_color))
            hex_color = self._rgb_to_hex(rgb)
            
            # Classify shape
            shape_type = self._classify_shape(contour, w, h)
            if not shape_type:
                continue
            
            swatch = {
                'type': shape_type,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': int(area),
                'color_rgb': rgb,
                'color_hex': hex_color
            }
            swatches.append(swatch)
        
        return swatches

    def _classify_shape(self, contour, w, h):
        """Classify shape"""
        if w == 0 or h == 0:
            return None
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 'rectangle'
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        aspect_ratio = float(w) / h
        
        # Circle
        if circularity > 0.75:
            return 'circle'
        
        # Square or Rectangle
        if num_vertices == 4:
            if 0.75 < aspect_ratio < 1.25:
                return 'square'
            else:
                return 'rectangle'
        
        # Other polygon shapes
        if 3 <= num_vertices <= 8:
            return 'rectangle'
        
        return None

    def _remove_duplicates(self, swatches):
        """Remove swatches that are very similar"""
        if not swatches:
            return []
        
        unique = []
        for swatch in swatches:
            is_duplicate = False
            
            for existing in unique:
                # Check color similarity
                color_diff = sum(abs(a - b) for a, b in zip(swatch['color_rgb'], existing['color_rgb']))
                
                # Check if regions overlap
                s_rect = (swatch['x'], swatch['y'], swatch['x'] + swatch['width'], swatch['y'] + swatch['height'])
                e_rect = (existing['x'], existing['y'], existing['x'] + existing['width'], existing['y'] + existing['height'])
                
                overlap = not (s_rect[2] < e_rect[0] or s_rect[0] > e_rect[2] or 
                             s_rect[3] < e_rect[1] or s_rect[1] > e_rect[3])
                
                if color_diff < 35 and overlap:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(swatch)
        
        return unique

    def _remove_noise(self, swatches):
        """Remove very small or irregular swatches"""
        filtered = []
        for swatch in swatches:
            # Skip very small swatches (noise)
            if swatch['area'] < 1000:
                continue
            
            # Prefer rectangles/circles/squares (skip weird shapes)
            if swatch['type'] not in ['rectangle', 'circle', 'square']:
                continue
            
            filtered.append(swatch)
        
        return filtered

    def _rgb_to_hex(self, rgb):
        """Convert RGB tuple to hex string"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/detect-swatches', methods=['POST', 'OPTIONS'])
def detect_swatches():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image_data = data['image']
        min_swatch_area = data.get('min_swatch_area', 1500)

        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_array = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        detector = SwatchDetector(image_array, min_swatch_area)
        swatches = detector.detect_swatches()

        return jsonify({
            'success': True,
            'swatches': swatches,
            'count': len(swatches)
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
