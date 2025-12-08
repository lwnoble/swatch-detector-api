from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)


class SwatchDetector:
    def __init__(self, image_array, min_swatch_area=800):
        self.image = image_array
        self.height, self.width = self.image.shape[:2]
        self.min_swatch_area = min_swatch_area
        self.swatches = []

    def detect_swatches(self):
        """Detect color swatches - prioritize small, isolated blocks"""
        # Strategy: Find all uniform color regions, then filter for swatch-like properties
        swatches = self._find_uniform_regions()
        
        # Filter to keep only swatch-like regions (small, clean, isolated)
        swatches = self._filter_swatches(swatches)
        
        self.swatches = swatches
        self.swatches.sort(key=lambda s: s['area'], reverse=True)
        
        return self.swatches

    def _find_uniform_regions(self):
        """Find all regions with relatively uniform color"""
        # Convert image to LAB color space for better color clustering
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        # Quantize to fewer colors
        data = lab.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 25, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(lab.shape)
        
        # Convert back to BGR
        quantized_bgr = cv2.cvtColor(quantized, cv2.COLOR_LAB2BGR)
        
        # Find contours
        gray = cv2.cvtColor(quantized_bgr, cv2.COLOR_BGR2GRAY)
        
        # Use threshold instead of Canny for cleaner regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Get more regions (we'll filter later)
            if area < 300:
                continue
            if area > self.width * self.height * 0.7:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get actual color from original image
            region = self.image[y:y+h, x:x+w]
            if region.size == 0:
                continue
            
            avg_color = np.mean(region, axis=(0, 1))
            rgb = tuple(int(c) for c in reversed(avg_color))
            hex_color = self._rgb_to_hex(rgb)
            
            # Check how uniform this region is
            uniformity = self._check_uniformity(region)
            
            shape_type = self._classify_shape(contour, w, h)
            if not shape_type:
                continue
            
            regions.append({
                'type': shape_type,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': int(area),
                'color_rgb': rgb,
                'color_hex': hex_color,
                'uniformity': uniformity,
                'aspect_ratio': w / h if h > 0 else 1
            })
        
        return regions

    def _check_uniformity(self, region):
        """Rate how uniform a region is (0-100, higher = more uniform)"""
        if region.size == 0:
            return 0
        
        # Calculate standard deviation
        std_dev = np.std(region)
        
        # Normalize to 0-100 (lower std dev = higher uniformity)
        uniformity = max(0, 100 - (std_dev * 2))
        return uniformity

    def _filter_swatches(self, regions):
        """Filter regions to keep only swatch-like ones"""
        filtered = []
        
        for region in regions:
            # Swatches should be:
            # 1. Relatively uniform color
            if region['uniformity'] < 50:
                continue
            
            # 2. Not too small, not too large
            if region['area'] < self.min_swatch_area:
                continue
            if region['area'] > self.width * self.height * 0.25:
                continue
            
            # 3. Reasonably square or rectangular (not super elongated)
            aspect = region['aspect_ratio']
            if aspect < 0.3 or aspect > 3.0:
                continue
            
            # 4. Should be one of the clean shapes
            if region['type'] not in ['rectangle', 'circle', 'square']:
                continue
            
            filtered.append(region)
        
        # Remove duplicates
        filtered = self._remove_duplicates(filtered)
        
        return filtered

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
            if 0.8 < aspect_ratio < 1.2:
                return 'square'
            else:
                return 'rectangle'
        
        # Other shapes
        if 3 <= num_vertices <= 8:
            return 'rectangle'
        
        return None

    def _remove_duplicates(self, regions):
        """Remove very similar regions"""
        if not regions:
            return []
        
        unique = []
        for region in regions:
            is_dup = False
            
            for existing in unique:
                # Color similarity (in LAB space would be better, but RGB works)
                color_diff = sum(abs(a - b) for a, b in zip(region['color_rgb'], existing['color_rgb']))
                
                # Spatial overlap
                r1_x1, r1_y1 = region['x'], region['y']
                r1_x2, r1_y2 = r1_x1 + region['width'], r1_y1 + region['height']
                
                r2_x1, r2_y1 = existing['x'], existing['y']
                r2_x2, r2_y2 = r2_x1 + existing['width'], r2_y1 + existing['height']
                
                overlap = not (r1_x2 < r2_x1 or r1_x1 > r2_x2 or r1_y2 < r2_y1 or r1_y1 > r2_y2)
                
                if color_diff < 30 and overlap:
                    is_dup = True
                    break
            
            if not is_dup:
                # Remove uniformity from output
                clean_region = {k: v for k, v in region.items() if k not in ['uniformity', 'aspect_ratio']}
                unique.append(clean_region)
        
        return unique

    def _rgb_to_hex(self, rgb):
        """Convert RGB to hex"""
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
        min_swatch_area = data.get('min_swatch_area', 800)

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
