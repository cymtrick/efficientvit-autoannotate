import cv2
import numpy as np
import json
from pathlib import Path
import torch

# Import EfficientViT-SAM components.
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model

def analyze_texture(roi):
    """
    Convert the ROI to grayscale and compute texture variance using the Laplacian.
    Raspberries typically have a high texture variance.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Tune the threshold as needed.
    return texture_score > 100

def detect_raspberries(image_path):
    """
    Read an image, apply HSV thresholding combined with morphological operations and texture analysis
    to detect raspberry-like regions. Returns:
      - detected_boxes: list of bounding boxes (x, y, w, h)
      - annotated_img: a copy of the image with drawn detection boxes.
    """
    # Read the image in BGR.
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_original = img.copy()
    
    # Convert image to HSV color space.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges for red-ish hues.
    color_ranges = [
        (np.array([0, 50, 50]), np.array([10, 255, 255])),
        (np.array([0, 30, 150]), np.array([20, 150, 255])),
        (np.array([160, 30, 50]), np.array([180, 255, 255]))
    ]

    # Combine masks for all ranges.
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Apply morphological operations to remove noise and close gaps.
    kernel = np.ones((20, 20), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find external contours in the mask.
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set area thresholds for valid raspberry candidates.
    min_area = 600  
    max_area = 17600  
    
    detected_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.6 < aspect_ratio < 1.8:
                # Expand the bounding box slightly for texture analysis.
                x1 = max(0, x - 5)
                y1 = max(0, y - 5)
                x2 = min(img.shape[1], x + w + 5)
                y2 = min(img.shape[0], y + h + 5)
                roi = img_original[y1:y2, x1:x2]
                if analyze_texture(roi):
                    detected_boxes.append((x, y, w, h))
    
    # Draw bounding boxes on a copy of the original image.
    annotated_img = img.copy()
    for x, y, w, h in detected_boxes:
        # Draw with a 20-pixel margin (green box).
        cv2.rectangle(annotated_img, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)
    
    return detected_boxes, annotated_img

def mask_to_polygons(binary_mask):
    """
    Convert a binary mask (H×W, dtype=uint8) to a list of polygons.
    Each polygon is represented as a list of (x, y) coordinates.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        polygon = cnt.reshape(-1, 2).tolist()
        polygons.append(polygon)
    return polygons

def main():
    input_dir = Path("images/")
    output_dir = Path("output_images")
    output_dir.mkdir(exist_ok=True)
    
    # Load the EfficientViT-SAM model once.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Choose your model variant, e.g., "efficientvit-sam-xl1".
    sam_model = create_efficientvit_sam_model("efficientvit-sam-xl1", pretrained=True).to(device)
    sam_predictor = EfficientViTSamPredictor(sam_model)
    
    # Process each image in the input directory.
    for image_path in input_dir.glob("*.png"):
        print(f"Processing image: {image_path.name}")
        try:
            # 1. Detect raspberries using texture and color.
            boxes, annotated_img = detect_raspberries(image_path)
            print(f"Detected {len(boxes)} candidate regions.")

            # 2. Prepare the image for segmentation.
            # SAM requires the image in RGB.
            img_bgr = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(img_rgb)
            
            segmentation_results = []
            
            # 3. For each detected bounding box, run segmentation.
            for idx, (x, y, w, h) in enumerate(boxes):
                # Convert (x, y, w, h) to (x0, y0, x1, y1).
                box_coords = np.array([x, y, x + w, y + h])
                masks, scores, _ = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_coords[None, :],  # shape: [1, 4]
                    multimask_output=False   # single best mask
                )
                if len(masks) == 0:
                    continue
                # Use the first mask.
                binary_mask = masks[0].astype(np.uint8)
                polygons = mask_to_polygons(binary_mask)
                
                segmentation_results.append({
                    "box_index": idx,
                    "box_xywh": (x, y, w, h),
                    "polygons": polygons
                })
                
                # 4. Draw the segmentation polygons on the annotated image (red lines).
                for poly in polygons:
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            
            # 5. Save segmentation results as JSON.
            json_path = output_dir / f"mask_polygons_{image_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(segmentation_results, f, indent=2)
            print(f"Saved segmentation polygons to {json_path}")
            
            # 6. Save the annotated image (with bounding boxes and segmentation overlays).
            annotated_output_path = output_dir / f"annotated_{image_path.name}"
            cv2.imwrite(str(annotated_output_path), annotated_img)
            print(f"Saved annotated image to {annotated_output_path}")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")

if __name__ == "__main__":
    main()
