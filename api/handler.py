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
        """Detect swatches - simplified to extract dominant colors only"""
        # Just extract the top 6 dominant colors as swatches
        # Then get 48 more palette colors
        
        print(f"🎨 Extracting colors from {self.width}x{self.height} image...")
        
        # Get top 6 colors
        top_colors = self._extract_dominant_colors(6)
        
        # Mark as swatches
        for color in top_colors:
            color['type'] = 'swatch'
        
        # Get 48 palette colors
        palette_colors = self._extract_dominant_colors(48)
        for color in palette_colors:
            color['type'] = 'palette'
        
        # Combine
        all_colors = top_colors + palette_colors
        self.swatches = all_colors
        
        print(f"✅ Returning {len(all_colors)} colors")
        return self.swatches

    def _find_uniform_blocks(self):
        """Find solid color blocks using K-means + connected components"""
        # Resize for faster K-means
        small_height = min(400, self.height)
        scale = small_height / self.height
        small_width = int(self.width * scale)
        small_image = cv2.resize(self.image, (small_width, small_height))
        
        # K-means clustering on resized image
        pixels = small_image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 20, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        reduced = centers[labels.flatten()].reshape(small_image.shape)
        
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
            
            # Scale area threshold back to original image size
            scaled_area = area / (scale * scale)
            
            if scaled_area < self.min_swatch_area:
                continue
            if scaled_area > self.width * self.height * 0.4:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Scale coordinates back to original image
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
            
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
                'area': int(scaled_area),
                'color_rgb': rgb,
                'color_hex': hex_color
            }
            swatches.append(swatch)
        
        return swatches

    def _find_organized_colors(self):
        """Look specifically for swatches in bottom/side regions of image"""
        swatches = []
        
        # Resize for faster K-means
        small_height = min(400, self.height)
        scale = small_height / self.height
        small_width = int(self.width * scale)
        small_image = cv2.resize(self.image, (small_width, small_height))
        
        # Check bottom 30% of resized image
        bottom_y = int(small_height * 0.7)
        bottom_region = small_image[bottom_y:, :]
        
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
                
                # Scale area back to original image
                scaled_area = area / (scale * scale)
                
                if scaled_area < self.min_swatch_area * 0.8:
                    continue
                if scaled_area > self.width * self.height * 0.3:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Scale coordinates back to full image
                x = int(x / scale)
                y = int((y + bottom_y) / scale)
                w = int(w / scale)
                h = int(h / scale)
                
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
        """Extract dominant colors from image - finds visually distinct colors"""
        try:
            if num_colors <= 0:
                return []
            
            print(f"🎨 Extracting {num_colors} colors...")
            
            # Resize for K-means speed
            small_height = min(400, self.height)
            scale = small_height / self.height
            small_width = int(self.width * scale)
            small_image = cv2.resize(self.image, (small_width, small_height))
            
            # K-means with enough clusters to capture variety
            pixels = small_image.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # Use more clusters to get better color variety
            num_clusters = min(num_colors * 5, 256)
            _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            label_counts = np.bincount(labels.flatten())
            sorted_indices = np.argsort(-label_counts)
            
            colors = []
            for center_idx in sorted_indices:
                if len(colors) >= num_colors:
                    break
                
                bgr_color = centers[center_idx]
                rgb = tuple(int(c) for c in reversed(bgr_color))
                
                # Skip ONLY pure white (all channels > 250)
                if rgb[0] > 250 and rgb[1] > 250 and rgb[2] > 250:
                    print(f"  ⊘ Skipping pure white: {rgb}")
                    continue
                
                # Skip very dark colors (near black) - optional
                # if rgb[0] < 15 and rgb[1] < 15 and rgb[2] < 15:
                #     continue
                
                hex_color = self._rgb_to_hex(rgb)
                
                # Check if too similar to existing colors
                is_duplicate = False
                for existing in colors:
                    color_diff = sum(abs(a - b) for a, b in zip(rgb, existing['color_rgb']))
                    if color_diff < 25:  # Reduced from 40 to capture more color variations
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    colors.append({
                        'type': 'palette',
                        'x': 0,
                        'y': 0,
                        'width': 0,
                        'height': 0,
                        'area': int(label_counts[center_idx]),
                        'color_rgb': list(rgb),
                        'color_hex': hex_color
                    })
                    print(f"  ✓ Added {hex_color} {rgb}")
            
            print(f"✅ Extracted {len(colors)} colors")
            return colors
            
        except Exception as e:
            print(f"❌ Error extracting colors: {e}")
            import traceback
            traceback.print_exc()
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
