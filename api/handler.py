from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)


class SwatchDetector:
    def __init__(self, image_array, min_swatch_area=1000):
        self.image = image_array
        self.height, self.width = self.image.shape[:2]
        self.min_swatch_area = min_swatch_area
        self.swatches = []

    def detect_swatches(self):
        # Method 1: Contour-based detection (for explicit color blocks)
        swatches_contour = self._detect_by_contours()
        
        # Method 2: Color clustering (for regions of similar color)
        swatches_cluster = self._detect_by_clustering()
        
        # Merge and deduplicate
        all_swatches = swatches_contour + swatches_cluster
        self.swatches = self._deduplicate_swatches(all_swatches)
        self.swatches.sort(key=lambda s: s['area'], reverse=True)
        
        return self.swatches

    def _detect_by_contours(self):
        """Detect swatches using edge detection and contours"""
        blurred = cv2.bilateralFilter(self.image, 9, 75, 75)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        swatches = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_swatch_area or area > self.width * self.height * 0.7:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if the region inside the contour is uniform color
            if self._is_uniform_color(x, y, w, h, threshold=30):
                shape_type = self._classify_shape(contour, w, h)
                if shape_type:
                    rgb = self._get_average_color(x, y, w, h)
                    hex_color = self._rgb_to_hex(rgb)
                    
                    swatches.append({
                        'type': shape_type,
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'area': int(area),
                        'color_rgb': rgb,
                        'color_hex': hex_color
                    })
        
        return swatches

    def _detect_by_clustering(self):
        """Detect swatches using K-means color clustering"""
        # Reshape image for clustering
        pixels = self.image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 8  # Number of color clusters
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reshape labels back to image
        labels = labels.reshape((self.height, self.width))
        centers = np.uint8(centers)
        
        swatches = []
        
        # For each cluster, find contours
        for cluster_id in range(k):
            # Create binary mask for this cluster
            mask = (labels == cluster_id).astype(np.uint8) * 255
            
            # Find contours in this cluster
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_swatch_area or area > self.width * self.height * 0.7:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                shape_type = self._classify_shape(contour, w, h)
                
                if shape_type:
                    color = centers[cluster_id]
                    rgb = tuple(int(c) for c in reversed(color))
                    hex_color = self._rgb_to_hex(rgb)
                    
                    swatches.append({
                        'type': shape_type,
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'area': int(area),
                        'color_rgb': rgb,
                        'color_hex': hex_color
                    })
        
        return swatches

    def _is_uniform_color(self, x, y, w, h, threshold=30):
        """Check if a region has uniform color"""
        region = self.image[y:y+h, x:x+w]
        if region.size == 0:
            return False
        
        # Calculate standard deviation for each channel
        std_dev = np.std(region, axis=(0, 1))
        
        # If all channels have low std dev, it's uniform
        return np.all(std_dev < threshold)

    def _get_average_color(self, x, y, w, h):
        """Get average color in a region"""
        region = self.image[y:y+h, x:x+w]
        mean_color = np.mean(region, axis=(0, 1))
        rgb = tuple(int(c) for c in reversed(mean_color))
        return rgb

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
            return 'square' if 0.7 < aspect_ratio < 1.3 else 'rectangle'
        if 3 <= num_vertices <= 5:
            if aspect_ratio > 0.3:  # More lenient for tall rectangles
                return 'rectangle'
        return None

    def _deduplicate_swatches(self, swatches):
        """Remove duplicate swatches (same color, overlapping regions)"""
        if not swatches:
            return []
        
        unique_swatches = []
        for swatch in swatches:
            # Check if this swatch is too similar to an existing one
            is_duplicate = False
            for existing in unique_swatches:
                # Same color?
                color_diff = sum(abs(a - b) for a, b in zip(swatch['color_rgb'], existing['color_rgb']))
                # Overlapping region?
                x_overlap = not (swatch['x'] + swatch['width'] < existing['x'] or swatch['x'] > existing['x'] + existing['width'])
                y_overlap = not (swatch['y'] + swatch['height'] < existing['y'] or swatch['y'] > existing['y'] + existing['height'])
                
                if color_diff < 50 and x_overlap and y_overlap:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_swatches.append(swatch)
        
        return unique_swatches

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
        min_swatch_area = data.get('min_swatch_area', 1000)

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
