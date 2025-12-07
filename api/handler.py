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
        """
        Simple approach: Find contiguous regions of uniform color.
        A swatch is simply a region where all pixels are roughly the same color.
        """
        # Use flood fill to find uniform color regions
        swatches = self._find_by_flood_fill()
        
        # Remove duplicates (same color in same region)
        swatches = self._remove_near_duplicates(swatches)
        
        self.swatches = swatches
        self.swatches.sort(key=lambda s: s['area'], reverse=True)
        
        return self.swatches

    def _find_by_flood_fill(self):
        """Find uniform color regions using flood fill algorithm"""
        visited = np.zeros((self.height, self.width), dtype=bool)
        swatches = []
        
        # Scan image systematically
        for y in range(0, self.height, 5):  # Sample every 5 pixels for speed
            for x in range(0, self.width, 5):
                if visited[y, x]:
                    continue
                
                # Try flood fill from this point
                seed_color = self.image[y, x]
                mask = self._flood_fill(seed_color, threshold=25)
                
                if mask is None:
                    visited[y, x] = True
                    continue
                
                visited[mask] = True
                
                # Get region bounds
                coords = np.argwhere(mask)
                if len(coords) < self.min_swatch_area:
                    continue
                
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                area = len(coords)
                
                # Skip if too large (probably not a swatch)
                if area > self.width * self.height * 0.4:
                    continue
                
                # Get average color
                region_pixels = self.image[mask]
                avg_color = np.mean(region_pixels, axis=0)
                rgb = tuple(int(c) for c in reversed(avg_color))
                hex_color = self._rgb_to_hex(rgb)
                
                # Classify shape
                shape_type = self._classify_region_shape(mask, w, h)
                
                swatch = {
                    'type': shape_type,
                    'x': int(x_min),
                    'y': int(y_min),
                    'width': int(w),
                    'height': int(h),
                    'area': int(area),
                    'color_rgb': rgb,
                    'color_hex': hex_color,
                    'mask': mask
                }
                swatches.append(swatch)
        
        return swatches

    def _flood_fill(self, seed_color, threshold=25):
        """
        Flood fill from a seed point to find connected uniform color region.
        Returns a boolean mask of the filled region.
        """
        visited = np.zeros((self.height, self.width), dtype=bool)
        mask = np.zeros((self.height, self.width), dtype=bool)
        
        # Find seed points (pixels similar to seed color)
        color_diff = np.sum(np.abs(self.image.astype(float) - seed_color.astype(float)), axis=2)
        seed_points = np.argwhere(color_diff < threshold)
        
        if len(seed_points) < 10:
            return None
        
        # Flood fill using BFS
        from collections import deque
        queue = deque(seed_points[:100])  # Start with first 100 similar pixels
        
        while queue:
            y, x = queue.popleft()
            
            if y < 0 or y >= self.height or x < 0 or x >= self.width:
                continue
            if visited[y, x]:
                continue
            
            pixel_color = self.image[y, x]
            diff = np.sum(np.abs(pixel_color.astype(float) - seed_color.astype(float)))
            
            if diff > threshold:
                continue
            
            visited[y, x] = True
            mask[y, x] = True
            
            # Add neighbors to queue
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width and not visited[ny, nx]:
                    queue.append((ny, nx))
        
        area = np.count_nonzero(mask)
        if area < self.min_swatch_area:
            return None
        
        return mask

    def _classify_region_shape(self, mask, w, h):
        """Classify shape of the region"""
        area = np.count_nonzero(mask)
        perimeter = self._estimate_perimeter(mask)
        
        if perimeter == 0:
            return 'rectangle'
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2)
        aspect_ratio = w / h if h > 0 else 1
        
        if circularity > 0.75:
            return 'circle'
        elif 0.7 < aspect_ratio < 1.3:
            return 'square'
        else:
            return 'rectangle'

    def _estimate_perimeter(self, mask):
        """Estimate perimeter of a region"""
        # Count edges
        edges = 0
        for y in range(self.height - 1):
            for x in range(self.width - 1):
                if mask[y, x] != mask[y, x + 1]:
                    edges += 1
                if mask[y, x] != mask[y + 1, x]:
                    edges += 1
        return edges / 2

    def _remove_near_duplicates(self, swatches):
        """Remove swatches that are very similar in color and location"""
        if not swatches:
            return []
        
        unique = []
        for swatch in swatches:
            is_duplicate = False
            
            for existing in unique:
                # Same color?
                color_diff = sum(abs(a - b) for a, b in zip(swatch['color_rgb'], existing['color_rgb']))
                
                # Same location? (overlapping)
                x1, y1, w1, h1 = swatch['x'], swatch['y'], swatch['width'], swatch['height']
                x2, y2, w2, h2 = existing['x'], existing['y'], existing['width'], existing['height']
                
                x_overlap = not (x1 + w1 < x2 or x1 > x2 + w2)
                y_overlap = not (y1 + h1 < y2 or y1 > y2 + h2)
                
                if color_diff < 30 and x_overlap and y_overlap:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
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

        # Remove mask from response (for JSON serialization)
        for swatch in swatches:
            if 'mask' in swatch:
                del swatch['mask']

        return jsonify({
            'success': True,
            'swatches': swatches,
            'count': len(swatches)
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
