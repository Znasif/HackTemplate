import cv2
import numpy as np
import json
import os
import argparse
from PIL import Image
import torch
from ultralytics import SAM
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Dict, List, Tuple
import colorsys
import math # Added for the new scoring logic

# --- SCRIPT INSTRUCTIONS ---
# 1. Install necessary libraries:
#    pip install torch torchvision torchaudio ultralytics opencv-python-headless Pillow transformers accelerate
#
# 2. Download a SAM model (e.g., sam_l.pt) from Ultralytics or Meta.
#
# 3. Update the paths in the `if __name__ == "__main__":` block at the bottom of the script:
#    - `sam_model_path`: Path to your downloaded SAM model file.
#    - `image_to_process`: Path to the thermostat image you want to process.
#    - `output_directory`: Folder where the results will be saved.
# ---

class ControlPanelRegionGenerator:
    """
    Generates region files for control panels using a two-phase approach:
    1. Isolate the main control panel from the image, performing a perspective transform.
    2. Analyze the isolated panel for its sub-components (buttons, screens, etc.).
    """
    def __init__(self, florence_model_name: str, sam_model_path: str):
        """
        Initializes the models and sets up the device.
        """
        print("Initializing models...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize Florence-2 for OCR
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            florence_model_name,
            trust_remote_code=True
        ).eval().to(self.device)
        self.florence_processor = AutoProcessor.from_pretrained(
            florence_model_name, trust_remote_code=True
        )
        print(f"✅ Florence-2 model loaded: {florence_model_name}")

        # Initialize SAM for segmentation
        self.sam = SAM(sam_model_path)
        print(f"✅ SAM model loaded: {sam_model_path}")

    # --- HELPER FUNCTIONS FOR FILTERING AND TRANSFORMATION ---

    def _is_button_like_mask(self, mask: np.ndarray, image_shape: Tuple) -> bool:
        """Filter to check if a mask corresponds to a plausible control panel element."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < 200: return False
        image_area = image_shape[0] * image_shape[1]
        if (area / image_area) > 0.95: return False # Allow large panels but not the whole image
        x, y, w, h = cv2.boundingRect(largest_contour)
        if w == 0 or h == 0: return False
        aspect_ratio = w / h
        if aspect_ratio > 10.0 or aspect_ratio < 0.1: return False
        return True

    def filter_contained_regions(self, segments: List[Dict], inclusion_threshold: float = 0.90) -> List[Dict]:
        """
        Filters out larger segments that fully contain smaller segments.
        This is better for hierarchical detections (e.g., a button inside a panel).
        """
        if not segments: return []
        num_segments = len(segments)
        discard_indices = set()
        for i in range(num_segments):
            for j in range(num_segments):
                if i == j: continue
                bbox_i = segments[i]['bbox']
                bbox_j = segments[j]['bbox']
                area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
                area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
                if area_i == 0 or area_j == 0: continue
                x1_inter = max(bbox_i[0], bbox_j[0])
                y1_inter = max(bbox_i[1], bbox_j[1])
                x2_inter = min(bbox_i[2], bbox_j[2])
                y2_inter = min(bbox_i[3], bbox_j[3])
                inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
                if inter_area == 0: continue
                smaller_area = min(area_i, area_j)
                if inter_area / smaller_area > inclusion_threshold:
                    if area_i > area_j: discard_indices.add(i)
                    else: discard_indices.add(j)
        return [seg for idx, seg in enumerate(segments) if idx not in discard_indices]

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Orders 4 points into a consistent top-left, top-right, bottom-right, bottom-left order."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _transform_and_crop(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Performs perspective transformation on the region defined by the mask."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return np.array([])
        contour = max(contours, key=cv2.contourArea)
        rect_pts = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect_pts).astype(int)
        ordered_box = self._order_points(box)
        (tl, tr, br, bl) = ordered_box
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))
        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered_box, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped
    
    def _find_main_panel(self, segments: List[Dict], image: np.ndarray, debug_dir: str) -> Dict:
        """Scores each segment based on area and text content to find the main control panel."""
        best_segment = None
        return segments[0]
        max_score = -1
        print("   Scoring candidate panels:")
        for i, segment in enumerate(segments):
            text = self.extract_text_florence(image, segment['bbox'], f"panel_eval_{i}", debug_dir)
            num_chars = len("".join(text.split())) # Count non-whitespace chars
            area = np.sum(segment['mask'])
            if area < 5000: score = 0
            else: score = math.log(area + 1) * math.log(num_chars + 2) # +2 to avoid log(1)=0
            print(f"     - Candidate {i}: Area={int(area)}, Chars={num_chars}, Score={score:.2f}")
            if score > max_score:
                max_score = score
                best_segment = segment
        return best_segment

    # --- CORE PROCESSING AND OCR FUNCTIONS ---

    def segment_control_panel(self, image: np.ndarray) -> List[Dict]:
        """
        Use SAM to automatically segment an image into potential regions.
        The first element of the returned list is always a synthetic
        segment representing the entire image.
        """
        # Get image dimensions
        h, w, _ = image.shape

        # 1. Create a synthetic segment for the entire image
        full_image_mask = np.ones((h, w), dtype=np.uint8)
        full_image_bbox = [0, 0, w, h]
        full_image_segment = {
            'mask': full_image_mask,
            'bbox': full_image_bbox,
            'segment_id': -1,  # Use a special ID to identify it
            'confidence': 1.0  # Assign maximum confidence
        }

        # 2. Initialize the segments list with the full image segment as the first element
        segments = [full_image_segment]

        # 3. Now, run SAM and append its findings to the list
        results = self.sam(image, points=None, bboxes=None)
        
        if results and results[0].masks is not None:
            for i, mask in enumerate(results[0].masks.data):
                mask_np = mask.cpu().numpy()
                # Ensure the mask is not the entire image, as we've already added it
                if np.all(mask_np):
                    continue
                    
                if self._is_button_like_mask(mask_np, image.shape):
                    bbox = self._mask_to_bbox(mask_np)
                    segments.append({
                        'mask': mask_np,
                        'bbox': bbox,
                        'segment_id': i,
                        'confidence': results[0].masks.conf[i].item() if hasattr(results[0].masks, 'conf') and results[0].masks.conf is not None else 0.9
                    })
                    
        return segments

    def extract_text_florence(self, image: np.ndarray, bbox: List[int], region_idx: int, debug_dir: str):
        """Extracts text from a raw color crop without preprocessing."""
        try:
            h, w, _ = image.shape
            x1, y1, x2, y2 = bbox
            raw_region_crop = image[y1:y2, x1:x2]
            if raw_region_crop.size == 0: return ''
            raw_crop_path = os.path.join(debug_dir, f"region_{region_idx}_raw_for_ocr.png")
            cv2.imwrite(raw_crop_path, raw_region_crop)
            pil_image = Image.fromarray(cv2.cvtColor(raw_region_crop, cv2.COLOR_BGR2RGB))
            task_prompt = '<OCR>'
            inputs = self.florence_processor(text=task_prompt, images=pil_image, return_tensors="pt").to(self.device)
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=1024, num_beams=3, early_stopping=False, do_sample=False,
            )
            generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.florence_processor.post_process_generation(generated_text, task=task_prompt, image_size=(pil_image.width, pil_image.height))
            return parsed_answer.get(task_prompt, '').strip()
        except Exception as e:
            print(f"      ⚠️  Florence OCR failed for region {bbox}: {e}")
            return ''

    def process_control_panel(self, image_path: str, output_dir: str):
        """
        Main processing pipeline with a two-step approach:
        1. Isolate the primary control panel.
        2. Analyze the isolated panel for its components.
        """
        if not os.path.exists(image_path): raise FileNotFoundError(f"Image not found at {image_path}")
        os.makedirs(output_dir, exist_ok=True)
        panel_name = os.path.splitext(os.path.basename(image_path))[0]
        full_image = cv2.imread(image_path)
        print(f"\n🎮 Processing control panel: {panel_name}")
        debug_dir = os.path.join(output_dir, "debug_regions")
        os.makedirs(debug_dir, exist_ok=True)

        # --- PHASE 1: Find and Isolate the Main Control Panel ---
        print("\n--- PHASE 1: ISOLATING MAIN PANEL ---")
        print("1. Segmenting full image to find panel candidates...")
        initial_segments = self.segment_control_panel(full_image)
        print(f"   Found {len(initial_segments)} initial candidates.")
        print("2. Finding the best candidate based on size and text content...")
        main_panel_segment = self._find_main_panel(initial_segments, full_image, debug_dir)
        if main_panel_segment is None:
            print("❌ Could not identify a main control panel. Aborting.")
            return None
        print(f"   ✅ Best panel selected (Segment ID: {main_panel_segment['segment_id']}).")
        print("3. Applying perspective transform and cropping...")
        panel_image = self._transform_and_crop(full_image, main_panel_segment['mask'])
        if panel_image.size == 0:
            print("❌ Failed to transform the panel. Using a simple crop as fallback.")
            x1, y1, x2, y2 = main_panel_segment['bbox']
            panel_image = full_image[y1:y2, x1:x2]
        transformed_panel_path = os.path.join(output_dir, f"template.jpg")
        cv2.imwrite(transformed_panel_path, panel_image)
        print(f"   ✅ Saved isolated panel image to: {transformed_panel_path}")

        # --- PHASE 2: Analyze the Isolated Panel ---
        print("\n--- PHASE 2: ANALYZING ISOLATED PANEL ---")
        print("1. Segmenting the isolated panel for sub-regions...")
        panel_segments_raw = self.segment_control_panel(panel_image)
        print(f"   Found {len(panel_segments_raw)} raw sub-regions.")
        print("2. Filtering contained regions (e.g., panel containing buttons)...")
        panel_segments = self.filter_contained_regions(panel_segments_raw)
        panel_segments.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
        print(f"   Filtered down to {len(panel_segments)} specific sub-regions.")
        print("3. Extracting text for each final sub-region...")
        ocr_results = []
        for i, segment in enumerate(panel_segments):
            print(f"   - Processing sub-region {i+1}/{len(panel_segments)}...")
            ocr_result = self.extract_text_florence(panel_image, segment['bbox'], i, debug_dir)
            ocr_results.append(ocr_result)
            print(f"     -> Found text: '{ocr_result}'")

        # --- FINALIZATION ---
        print("\n4. Generating final output files...")
        colors = self.generate_unique_colors(len(panel_segments))
        regions_data = self.create_regions_json(panel_segments, colors, ocr_results)
        regions_path = os.path.join(output_dir, "regions.json")
        with open(regions_path, 'w') as f: json.dump(regions_data, f, indent=2)
        print(f"   ✅ Saved regions config: {regions_path}")
        color_map = self.create_color_map(panel_image, panel_segments, colors)
        colormap_path = os.path.join(output_dir, "colorMap.png")
        cv2.imwrite(colormap_path, color_map)
        print(f"   ✅ Saved color map: {colormap_path}")
        vis_image = self.create_visualization(panel_image, panel_segments, colors, ocr_results)
        vis_path = os.path.join(output_dir, f"{panel_name}_visualization.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"   ✅ Saved visualization: {vis_path}")
        print("\n🎉 Processing Complete!")
        return regions_data

    # --- OUTPUT GENERATION UTILITIES ---

    def generate_unique_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors

    def create_color_map(self, image: np.ndarray, segments: List[Dict], colors: List[Tuple[int, int, int]]) -> np.ndarray:
        color_map = np.zeros_like(image)
        for i, segment in enumerate(segments):
            mask = cv2.resize(segment['mask'].astype(np.uint8), (image.shape[1], image.shape[0]))
            color_map[mask > 0] = colors[i]
        return color_map

    def create_regions_json(self, segments: List[Dict], colors: List[Tuple[int, int, int]], ocr_results: List[Dict]) -> Dict:
        regions_data = {"regions": [], "metadata": {}}
        for i, (segment, color, text) in enumerate(zip(segments, colors, ocr_results)):
            class_name = text if text else f"region_{i+1}"
            regions_data["regions"].append({
                "detection_id": f"region_{i+1:03d}", "class": class_name, "text_content": text,
                "color": list(color), "bbox": segment['bbox'], "segment_area": int(np.sum(segment['mask'])),
                "center_point": self._get_mask_center(segment['mask'])
            })
        regions_data["metadata"] = {
            "total_regions": len(segments), "generation_method": "florence2_sam_2phase", "color_format": "bgr"
        }
        return regions_data

    def create_visualization(self, image: np.ndarray, segments: List[Dict], colors: List[Tuple[int, int, int]], ocr_results: List[Dict]) -> np.ndarray:
        vis_image = image.copy()
        overlay = vis_image.copy()
        for i, (segment, color, ocr) in enumerate(zip(segments, colors, ocr_results)):
            mask = cv2.resize(segment['mask'].astype(np.uint8), (image.shape[1], image.shape[0]))
            overlay[mask > 0] = color
        vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)
        for i, (segment, color, ocr) in enumerate(zip(segments, colors, ocr_results)):
            x1, y1, x2, y2 = segment['bbox']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            label = f"R{i+1}: {ocr}"
            cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return vis_image

    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols): return [0,0,0,0]
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return [int(xmin), int(ymin), int(xmax), int(ymax)]

    def _get_mask_center(self, mask: np.ndarray) -> List[int]:
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0: return [0,0]
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return [cx, cy]


if __name__ == "__main__":
    # --- COMMAND-LINE ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(
        description="Analyzes a control panel image to identify and OCR its components. "
                    "First, it isolates the main panel, then analyzes its sub-regions."
    )
    
    # Required argument for the image file
    parser.add_argument(
        "image_to_process",
        type=str,
        help="Path to the input image file to be processed."
    )
    
    # Optional arguments for other paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the results. If not provided, a new directory will be created next to the input image."
    )
    parser.add_argument(
        "--sam_model_path",
        type=str,
        default="../models/sam2.1_l.pt",
        help="Path to the SAM model file."
    )
    parser.add_argument(
        "--florence_model_name",
        type=str,
        default="microsoft/Florence-2-large",
        help="Name of the Florence-2 model from Hugging Face."
    )
    
    args = parser.parse_args()

    # --- SETTING UP PATHS ---
    image_to_process = "../models/cars/"+args.image_to_process
    sam_model_path = args.sam_model_path
    florence_model_name = args.florence_model_name

    # Determine the output directory
    if args.output_dir:
        output_directory = args.output_dir
    else:
        # Create a default output directory named after the image file
        base_name = os.path.splitext(os.path.basename(image_to_process))[0]
        output_directory = os.path.join(os.path.dirname(image_to_process), base_name + "_output")

    # --- RUN THE PROCESSING PIPELINE ---
    try:
        generator = ControlPanelRegionGenerator(
            florence_model_name=florence_model_name,
            sam_model_path=sam_model_path
        )
        
        final_regions = generator.process_control_panel(
            image_path=image_to_process,
            output_dir=output_directory
        )

        print("\n--- FINAL RESULTS SUMMARY ---")
        if final_regions and final_regions.get('regions'):
            for region in final_regions['regions']:
                print(f"  - ID: {region['detection_id']}, Text: '{region['text_content']}'")
        else:
            print("  No valid regions were detected after filtering.")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: A required file was not found.")
        print(f"   Details: {e}")
        print("   Please ensure all file paths are correct.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print(f"   Please check your library installations and model paths.")