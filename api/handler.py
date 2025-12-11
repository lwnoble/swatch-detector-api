from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)


class SwatchDetector:
    def __init__(self, image_array, min_swatch_area=400):
        self.image = image_array
        self.height, self.width = self.image.shape[:2]
        self.min_swatch_area = min_swatch_area
        self.swatches = []

    def detect_swatches(self):
        """Detect swatches using multiple strategies"""
        all_swatches = []
        
        # Strategy 1: Find uniform color blocks
        swatches_1 = self._find_uniform_blocks()
        all_swatches.extend(swatches_1)
        
        # Strategy 2: Look for organized color regions (grids/rows)
        swatches_2 = self._find_organized_colors()
        all_swatches.extend(swatches_2)
        
        # Remove duplicates
        unique_swatches = self._deduplicate(all_swatches)
        
        # Filter out white colors
        filtered_swatches = [s for s in unique_swatches if not self._is_white(s['color_rgb'])]
        
        # Mark all detected swatches with type='swatch'
        for swatch in filtered_swatches:
            swatch['type'] = 'swatch'
        
        # Always add palette colors to reach 10 total
        if len(filtered_swatches) < 10:
            palette_colors = self._extract_dominant_colors(10 - len(filtered_swatches))
            filtered_swatches.extend(palette_colors)
        
        # Limit to 10 colors max
        self.swatches = filtered_swatches[:10]
        self.swatches.sort(key=lambda s: s['area'], reverse=True)
        
        return self.swatches

    def _find_uniform_blocks(self):
        """Find solid color blocks using K-means + connected components"""
        # K-means clustering
        pixels = self.image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 20, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        reduced = centers[labels.flatten()].reshape(self.image.shape)
        
        # Convert to grayscale and find regions
        gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find distinct regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        swatches = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_swatch_area:
                continue
            if area > self.width * self.height * 0.4:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get actual color from original image - sample center for vibrant colors
            region = self.image[y:y+h, x:x+w]
            if region.size == 0:
                continue
            
            # Sample center pixel for most vibrant color
            center_y, center_x = h // 2, w // 2
            if 0 <= center_y < region.shape[0] and 0 <= center_x < region.shape[1]:
                center_color = region[center_y, center_x]
                rgb = tuple(int(c) for c in reversed(center_color))
            else:
                # Fallback to mean if center sampling fails
                avg_color = np.mean(region, axis=(0, 1))
                rgb = tuple(int(c) for c in reversed(avg_color))
            
            hex_color = self._rgb_to_hex(rgb)
            
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

    def _find_organized_colors(self):
        """Look specifically for swatches in bottom/side regions of image"""
        swatches = []
        
        # Check bottom 30% of image (common swatch location)
        bottom_y = int(self.height * 0.7)
        bottom_region = self.image[bottom_y:, :]
        
        # K-means on bottom region
        pixels = bottom_region.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        if len(pixels) < 100:
            return swatches
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, 15, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        except:
            return swatches
        
        centers = np.uint8(centers)
        
        # For each color cluster, find connected regions
        for cluster_id in range(len(centers)):
            cluster_mask = (labels.reshape(bottom_region.shape[:2]) == cluster_id).astype(np.uint8)
            
            # Find contours in this cluster
            contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < self.min_swatch_area * 0.8:
                    continue
                if area > self.width * self.height * 0.3:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Adjust y coordinate back to full image
                y += bottom_y
                
                region = self.image[y:y+h, x:x+w]
                if region.size == 0:
                    continue
                
                # Sample center pixel for vibrant colors
                center_y, center_x = h // 2, w // 2
                if 0 <= center_y < region.shape[0] and 0 <= center_x < region.shape[1]:
                    center_color = region[center_y, center_x]
                    rgb = tuple(int(c) for c in reversed(center_color))
                else:
                    avg_color = np.mean(region, axis=(0, 1))
                    rgb = tuple(int(c) for c in reversed(avg_color))
                
                hex_color = self._rgb_to_hex(rgb)
                
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
        aspect_ratio = w / h if h > 0 else 1
        
        # Circle
        if circularity > 0.75:
            return 'circle'
        
        # Square or Rectangle
        if num_vertices == 4:
            if 0.75 < aspect_ratio < 1.25:
                return 'square'
            else:
                return 'rectangle'
        
        if 3 <= num_vertices <= 8:
            return 'rectangle'
        
        return None

    def _deduplicate(self, swatches):
        """Remove duplicate swatches"""
        if not swatches:
            return []
        
        unique = []
        for swatch in swatches:
            is_dup = False
            
            for existing in unique:
                # Color similarity
                color_diff = sum(abs(a - b) for a, b in zip(swatch['color_rgb'], existing['color_rgb']))
                
                # Spatial overlap
                s_x1, s_y1 = swatch['x'], swatch['y']
                s_x2, s_y2 = s_x1 + swatch['width'], s_y1 + swatch['height']
                
                e_x1, e_y1 = existing['x'], existing['y']
                e_x2, e_y2 = e_x1 + existing['width'], e_y1 + existing['height']
                
                overlap = not (s_x2 < e_x1 or s_x1 > e_x2 or s_y2 < e_y1 or s_y1 > e_y2)
                
                if color_diff < 25 and overlap:
                    is_dup = True
                    break
            
            if not is_dup:
                unique.append(swatch)
        
        return unique

    def _rgb_to_hex(self, rgb):
        """Convert RGB to hex"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    def _is_white(self, rgb):
        """Check if color is white (R, G, B all > 240)"""
        return rgb[0] > 240 and rgb[1] > 240 and rgb[2] > 240

    def _extract_dominant_colors(self, num_colors=10):
        """Extract dominant colors from image as fallback palette"""
        try:
            if num_colors <= 0:
                return []
            
            # Resize image for faster processing
            small_height = min(200, self.height)
            scale = small_height / self.height
            small_width = int(self.width * scale)
            small_image = cv2.resize(self.image, (small_width, small_height))
            
            # K-means clustering - search through many clusters to find diverse colors
            pixels = small_image.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            num_clusters = min(num_colors * 3, 80)  # Search through more clusters for diversity
            _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            label_counts = np.bincount(labels.flatten())
            sorted_indices = np.argsort(-label_counts)
            
            swatches = []
            for center_idx in sorted_indices:
                bgr_color = centers[center_idx]
                rgb = tuple(int(c) for c in reversed(bgr_color))
                
                # Skip white colors
                if self._is_white(rgb):
                    continue
                
                hex_color = self._rgb_to_hex(rgb)
                
                # Avoid duplicate colors (within palette only, not swatches)
                is_duplicate = False
                for existing in swatches:
                    color_diff = sum(abs(int(a) - int(b)) for a, b in zip(rgb, existing['color_rgb']))
                    if color_diff < 30:
                        is_duplicate = True
                        break
                
                if not is_duplicate and len(swatches) < num_colors:
                    swatches.append({
                        'type': 'palette',
                        'x': 0,
                        'y': 0,
                        'width': 0,
                        'height': 0,
                        'area': int(label_counts[center_idx]),
                        'color_rgb': rgb,
                        'color_hex': hex_color
                    })
                
                if len(swatches) >= num_colors:
                    break
            
            return swatches
        except Exception as e:
            print(f"Error extracting dominant colors: {e}")
            return []


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
        min_swatch_area = data.get('min_swatch_area', 400)

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
