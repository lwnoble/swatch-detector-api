from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
from collections import deque

app = Flask(__name__)


class SwatchDetector:
    def __init__(self, image_array, min_swatch_area=1500):
        self.image = image_array
        self.height, self.width = self.image.shape[:2]
        self.min_swatch_area = min_swatch_area
        self.swatches = []

    def detect_swatches(self):
        """Find uniform color regions efficiently"""
        # Downsample for faster processing
        small_image = cv2.resize(self.image, (self.width // 2, self.height // 2))
        
        # Reduce colors to find distinct regions
        reduced = self._reduce_colors(small_image, num_colors=15)
        
        # Find contours in reduced image
        swatches = self._find_contours_in_reduced(reduced)
        
        # Scale coordinates back up
        for swatch in swatches:
            swatch['x'] *= 2
            swatch['y'] *= 2
            swatch['width'] *= 2
            swatch['height'] *= 2
            swatch['area'] *= 4
        
        # Remove duplicates
        swatches = self._remove_duplicates(swatches)
        
        self.swatches = swatches
        self.swatches.sort(key=lambda s: s['area'], reverse=True)
        
        return self.swatches

    def _reduce_colors(self, image, num_colors=15):
        """Reduce image to dominant colors using K-means"""
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        return result.reshape(image.shape)

    def _find_contours_in_reduced(self, reduced_image):
        """Find contours in the reduced color image"""
        # Convert to grayscale and find edges
        gray = cv2.cvtColor(reduced_image, cv2.COLOR_BGR2GRAY)
        
        # Use morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        swatches = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_swatch_area / 4:  # Scaled down
                continue
            if area > (self.width // 2) * (self.height // 2) * 0.5:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get color from this region
            region = reduced_image[y:y+h, x:x+w]
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
        """Classify shape of the region"""
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
        
        # Allow other shapes
        if 3 <= num_vertices <= 6:
            return 'rectangle'
        
        return None

    def _remove_duplicates(self, swatches):
        """Remove very similar swatches"""
        if not swatches:
            return []
        
        unique = []
        for swatch in swatches:
            is_dup = False
            
            for existing in unique:
                # Color similarity
                color_diff = sum(abs(a - b) for a, b in zip(swatch['color_rgb'], existing['color_rgb']))
                
                # Location overlap
                x1, y1, w1, h1 = swatch['x'], swatch['y'], swatch['width'], swatch['height']
                x2, y2, w2, h2 = existing['x'], existing['y'], existing['width'], existing['height']
                
                x_overlap = not (x1 + w1 < x2 or x1 > x2 + w2)
                y_overlap = not (y1 + h1 < y2 or y1 > y2 + h2)
                
                if color_diff < 40 and x_overlap and y_overlap:
                    is_dup = True
                    break
            
            if not is_dup:
                unique.append(swatch)
        
        return unique

    def _rgb_to_hex(self, rgb):
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
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
