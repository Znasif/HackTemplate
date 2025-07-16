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
import math

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
    
    Also includes a method for processing structured images like Magic: The Gathering cards.
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

    # --- HELPER FUNCTIONS FOR FILTERING AND TRANSFORMATION (for Control Panels) ---

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
        max_score = -1
        print("   Scoring candidate panels:")
        for i, segment in enumerate(segments):
            # We don't need to re-run OCR for the entire image if it's the first synthetic segment
            if segment.get('segment_id') == -1:
                continue

            text = self.extract_text_florence(image, segment['bbox'], f"panel_eval_{i}", debug_dir)
            num_chars = len("".join(text.split())) # Count non-whitespace chars
            area = np.sum(segment['mask'])
            
            # Simple heuristic: penalize very small areas
            if area < 5000: 
                score = 0
            else: 
                score = math.log(area + 1) * math.log(num_chars + 2) # +2 to avoid log(1)=0
            
            print(f"     - Candidate {i}: Area={int(area)}, Chars={num_chars}, Score={score:.2f}")

            if score > max_score:
                max_score = score
                best_segment = segment
                
        # Fallback to the largest segment if scoring fails to find a good candidate
        if best_segment is None:
            print("   ⚠️ Scoring did not find a text-rich panel. Falling back to the largest segment.")
            # Exclude the full-image synthetic segment from the fallback search
            valid_segments = [s for s in segments if s.get('segment_id') != -1]
            if valid_segments:
                best_segment = max(valid_segments, key=lambda s: np.sum(s['mask']))
            else: # If only the synthetic segment exists
                best_segment = segments[0]
                
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

        # 2. Initialize the segments list with the full image segment
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

    def extract_text_florence(self, image: np.ndarray, bbox: List[int], region_idx, debug_dir: str):
        """Extracts text from a raw color crop using Florence-2."""
        try:
            h, w, _ = image.shape
            x1, y1, x2, y2 = bbox
            # Add a small padding to the crop to ensure text isn't cut off at the edges
            padding = 2
            raw_region_crop = image[max(0, y1-padding):min(h, y2+padding), max(0, x1-padding):min(w, x2+padding)]
            
            if raw_region_crop.size == 0: return ''
            
            # Save the crop for debugging purposes
            os.makedirs(debug_dir, exist_ok=True)
            raw_crop_path = os.path.join(debug_dir, f"region_{region_idx}_for_ocr.png")
            cv2.imwrite(raw_crop_path, raw_region_crop)

            pil_image = Image.fromarray(cv2.cvtColor(raw_region_crop, cv2.COLOR_BGR2RGB))
            task_prompt = '<OCR>'
            inputs = self.florence_processor(text=task_prompt, images=pil_image, return_tensors="pt").to(self.device)
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=1024, num_beams=3, early_stopping=False, do_sample=False,
            )
            generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            # Use post_process_generation for better parsing
            parsed_answer = self.florence_processor.post_process_generation(generated_text, task=task_prompt, image_size=(pil_image.width, pil_image.height))
            # Clean up the final text
            final_text = parsed_answer.get(task_prompt, '').strip()
            # Replace newline characters with spaces for cleaner single-line output
            return " ".join(final_text.splitlines())

        except Exception as e:
            print(f"      ⚠️  Florence OCR failed for region {bbox}: {e}")
            return ''

    # --- NEW: MTG CARD PROCESSING FUNCTIONS ---
    
    def _get_mtg_card_layout_boxes(self, image_shape: Tuple) -> List[Dict]:
        """
        Defines the standard regions of an MTG card using percentage-based bounding boxes.
        This heuristic-based approach is more reliable for structured layouts than open-ended segmentation.
        """
        h, w = image_shape[:2]
        
        # Percentages are tuned for a standard, modern MTG card frame.
        # Format: [x_start, y_start, x_end, y_end] as fractions of width/height
        region_definitions = {
            "Title":       [0.08, 0.05, 0.65, 0.10],
            "Type Line":   [0.08, 0.57, 0.92, 0.62],
            "Rules Text":  [0.08, 0.64, 0.92, 0.85],
            "Flavor Text": [0.08, 0.85, 0.92, 0.91],
        }
        
        segments = []
        for label, coords in region_definitions.items():
            x1 = int(coords[0] * w)
            y1 = int(coords[1] * h)
            x2 = int(coords[2] * w)
            y2 = int(coords[3] * h)
            
            bbox = [x1, y1, x2, y2]
            
            # Create a corresponding mask for compatibility with visualization functions
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            segments.append({
                'label': label,
                'bbox': bbox,
                'mask': mask
            })
            
        return segments

    def process_mtg_card(self, image_path: str, output_dir: str):
        """
        Main processing pipeline for MTG cards. It defines regions based on a standard
        card layout and then uses OCR to extract text from each one.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        card_name = os.path.splitext(os.path.basename(image_path))[0]
        full_image = cv2.imread(image_path)
        print(f"\n🃏 Processing MTG Card: {card_name}")
        debug_dir = os.path.join(output_dir, "debug_regions")
        os.makedirs(debug_dir, exist_ok=True)

        # --- PHASE 1: Define Card Regions ---
        print("\n--- PHASE 1: DEFINING CARD REGIONS ---")
        print("1. Defining regions based on standard MTG card layout...")
        card_segments = self._get_mtg_card_layout_boxes(full_image.shape)
        card_segments.sort(key=lambda s: s['bbox'][1]) # Sort top-to-bottom
        print(f"   ✅ Defined {len(card_segments)} key regions: {[s['label'] for s in card_segments]}")
        
        # Save the full card image as the base "template"
        template_path = os.path.join(output_dir, f"template.jpg")
        cv2.imwrite(template_path, full_image)
        print(f"   ✅ Saved base card image to: {template_path}")

        # --- PHASE 2: Analyze Defined Regions ---
        print("\n--- PHASE 2: ANALYZING REGIONS ---")
        print("1. Extracting text for each defined region...")
        ocr_results = []
        for i, segment in enumerate(card_segments):
            label = segment['label']
            print(f"   - Processing region {i+1}/{len(card_segments)}: '{label}'...")
            
            text = self.extract_text_florence(full_image, segment['bbox'], label, debug_dir)
            ocr_results.append(text)
            print(f"     -> Found text: '{text}'")

        # --- FINALIZATION ---
        print("\n2. Generating final output files...")
        colors = self.generate_unique_colors(len(card_segments))
        
        # The existing output functions are reused here. We format our data to match their expectations.
        # We modify the 'segments' to add the text to the class name for the JSON.
        for seg, text in zip(card_segments, ocr_results):
            seg['class_name'] = seg['label']
            seg['text_content'] = text
        
        regions_data = self.create_regions_json(card_segments, colors, ocr_results)
        regions_path = os.path.join(output_dir, "regions.json")
        with open(regions_path, 'w') as f:
            json.dump(regions_data, f, indent=2)
        print(f"   ✅ Saved regions config: {regions_path}")
        
        color_map = self.create_color_map(full_image, card_segments, colors)
        colormap_path = os.path.join(output_dir, "colorMap.png")
        cv2.imwrite(colormap_path, color_map)
        print(f"   ✅ Saved color map: {colormap_path}")
        
        vis_image = self.create_visualization(full_image, card_segments, colors, ocr_results)
        vis_path = os.path.join(output_dir, f"{card_name}_visualization.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"   ✅ Saved visualization: {vis_path}")
        
        print("\n🎉 Processing Complete!")
        return regions_data

    # --- EXISTING CONTROL PANEL PROCESSING FUNCTION (UNCHANGED) ---

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

    # --- OUTPUT GENERATION UTILITIES (UNCHANGED) ---

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
            # Ensure mask has the same dimensions as the image
            if segment['mask'].shape != image.shape[:2]:
                mask = cv2.resize(segment['mask'].astype(np.uint8), (image.shape[1], image.shape[0]))
            else:
                mask = segment['mask']
            color_map[mask > 0] = colors[i]
        return color_map

    def create_regions_json(self, segments: List[Dict], colors: List[Tuple[int, int, int]], ocr_results: List[str]) -> Dict:
        regions_data = {"regions": [], "metadata": {}}
        for i, (segment, color, text) in enumerate(zip(segments, colors, ocr_results)):
            # Use a more descriptive class name from the segment if available, otherwise default
            class_name = segment.get('label', f"region_{i+1}")
            regions_data["regions"].append({
                "detection_id": f"region_{i+1:03d}", "class": class_name, "text_content": text,
                "color": list(color), "bbox": segment['bbox'], "segment_area": int(np.sum(segment['mask'])),
                "center_point": self._get_mask_center(segment['mask'])
            })
        regions_data["metadata"] = {
            "total_regions": len(segments), "generation_method": "florence2_sam_2phase", "color_format": "bgr"
        }
        return regions_data

    def create_visualization(self, image: np.ndarray, segments: List[Dict], colors: List[Tuple[int, int, int]], ocr_results: List[str]) -> np.ndarray:
        vis_image = image.copy()
        overlay = vis_image.copy()
        for i, (segment, color) in enumerate(zip(segments, colors)):
            if segment['mask'].shape != image.shape[:2]:
                mask = cv2.resize(segment['mask'].astype(np.uint8), (image.shape[1], image.shape[0]))
            else:
                mask = segment['mask']
            overlay[mask > 0] = color
        vis_image = cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0)
        
        for i, (segment, color, ocr) in enumerate(zip(segments, colors, ocr_results)):
            x1, y1, x2, y2 = segment['bbox']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            label = segment.get('label', f"R{i+1}")
            cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
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
    parser = argparse.ArgumentParser(
        description="Analyzes a control panel or MTG card image to identify and OCR its components."
    )
    
    parser.add_argument("image_to_process", type=str, help="Path to the input image file.")
    parser.add_argument("--mode", type=str, default="control_panel", choices=["control_panel", "mtg_card"], help="Processing mode.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results.")
    parser.add_argument("--sam_model_path", type=str, default="../models/sam2.1_l.pt", help="Path to the SAM model.")
    parser.add_argument("--florence_model_name", type=str, default="microsoft/Florence-2-large", help="Florence-2 model name.")
    
    args = parser.parse_args()

    # --- SETTING UP PATHS ---
    image_to_process = "../models/cars/"+args.image_to_process
    if not os.path.exists(image_to_process):
        print(f"❌ Error: Input image not found at '{image_to_process}'")
        exit()
        
    if args.output_dir:
        output_directory = args.output_dir
    else:
        base_name = os.path.splitext(os.path.basename(image_to_process))[0]
        output_directory = os.path.join(os.path.dirname(image_to_process), base_name + "_output")

    # --- RUN THE PROCESSING PIPELINE ---
    try:
        generator = ControlPanelRegionGenerator(
            florence_model_name=args.florence_model_name,
            sam_model_path=args.sam_model_path
        )
        
        if args.mode == "control_panel":
            final_regions = generator.process_control_panel(
                image_path=image_to_process,
                output_dir=output_directory
            )
        elif args.mode == "mtg_card":
            final_regions = generator.process_mtg_card(
                image_path=image_to_process,
                output_dir=output_directory
            )

        print("\n--- FINAL RESULTS SUMMARY ---")
        if final_regions and final_regions.get('regions'):
            for region in final_regions['regions']:
                print(f"  - Class: {region['class']}, Text: '{region['text_content']}'")
        else:
            print("  No valid regions were detected or processed.")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: A required file was not found.")
        print(f"   Details: {e}")
    except Exception as e:
        import traceback
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()