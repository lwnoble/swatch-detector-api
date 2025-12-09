from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
import traceback

app = Flask(__name__)

class SwatchDetector:
    def __init__(self, image, min_area=400):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.min_area = min_area

    def detect(self):
        pixels = self.image.reshape((-1, 3))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 20, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        reduced = centers[labels.flatten()].reshape(self.image.shape)
        gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        swatches = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.width * self.height * 0.4:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            region = self.image[y:y+h, x:x+w]
            if region.size == 0:
                continue
            center_y, center_x = h // 2, w // 2
            if 0 <= center_y < region.shape[0] and 0 <= center_x < region.shape[1]:
                color = region[center_y, center_x]
                rgb = tuple(int(c) for c in reversed(color))
            else:
                avg = np.mean(region, axis=(0, 1))
                rgb = tuple(int(c) for c in reversed(avg))
            hex_color = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
            shape = self._get_shape(contour, w, h)
            if not shape:
                continue
            swatches.append({
                'type': 'swatch',
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': int(area),
                'color_rgb': rgb,
                'color_hex': hex_color
            })
        
        # Fallback: if we found less than 3 swatches, extract dominant colors
        if len(swatches) < 3:
            try:
                dominant_colors = self.extract_dominant_colors(10 - len(swatches))
                swatches.extend(dominant_colors)
            except Exception as e:
                print(f"Warning: Could not extract dominant colors: {e}")
        
        return swatches

    def extract_dominant_colors(self, num_colors=10):
        """Extract the most common colors from the image (fallback method)"""
        try:
            if num_colors <= 0:
                return []
            
            # Reduce image to reasonable size
            small_height = min(200, self.height)
            scale = small_height / self.height
            small_width = int(self.width * scale)
            small_image = cv2.resize(self.image, (small_width, small_height))
            
            pixels = small_image.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            num_clusters = min(num_colors + 5, 50)
            _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            label_counts = np.bincount(labels.flatten())
            sorted_indices = np.argsort(-label_counts)
            
            swatches = []
            for center_idx in sorted_indices[:num_clusters]:
                bgr_color = centers[center_idx]
                rgb = tuple(int(c) for c in reversed(bgr_color))
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                
                # Avoid duplicates
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
            
            return swatches
        except Exception as e:
            print(f"Error in extract_dominant_colors(): {e}")
            traceback.print_exc()
            return []

    def _get_shape(self, contour, w, h):
        if w == 0 or h == 0:
            return None
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 'rectangle'
        circularity = 4 * np.pi * area / (perimeter ** 2)
        aspect = w / h if h > 0 else 1
        if circularity > 0.75:
            return 'circle'
        if 0.75 < aspect < 1.25:
            return 'square'
        return 'rectangle'

@app.after_request
def cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/health', methods=['GET'])
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
        min_area = data.get('min_swatch_area', 400)
        
        # Handle data URI format
        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid base64: {str(e)}'}), 400
        
        # Load image
        try:
            image_pil = Image.open(io.BytesIO(image_bytes))
            image_array = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid image: {str(e)}'}), 400
        
        # Detect swatches
        try:
            detector = SwatchDetector(image_array, min_area)
            swatches = detector.detect()
        except Exception as e:
            print(f"Detection error: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Detection failed: {str(e)}'}), 500
        
        return jsonify({'success': True, 'swatches': swatches, 'count': len(swatches)})
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run()
