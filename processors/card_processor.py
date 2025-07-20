import cv2
import numpy as np
import os
import time
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from typing import Dict, Tuple, List, Optional, Union
from ultralytics import YOLO
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import json
import urllib.parse
import re

# Assuming BaseProcessor is defined elsewhere
from .base_processor import BaseProcessor

class RealTimeCardProcessor(BaseProcessor):
    """
    A streamlined processor that uses YOLO instance segmentation for card detection,
    filters by aspect ratio, and performs OCR on the largest matching segment.
    Runs detection, warping, and OCR in parallel for better performance.
    """
    # --- Configuration Constants ---
    MIN_CONTOUR_AREA = 5000
    ASPECT_RATIO_TOLERANCE = 0.25  # Increased tolerance for non-frontoparallel cards
    MAX_CORNERS = 8  # Allow more complex contours

    def __init__(self, ocr_model_name: str, yolo_model_path: str = "yolov8n-seg.pt"):
        super().__init__()
        print("Initializing RealTimeCardProcessor (Parallel YOLO-Segmentation Version)...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # --- OCR Model Initialization ---
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_name).eval().to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(ocr_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(ocr_model_name)
        print(f"✅ TrOCR model and components loaded: {ocr_model_name}")

        # --- YOLO Model Initialization ---
        self.yolo_model = YOLO(yolo_model_path)
        print(f"✅ YOLO segmentation model loaded: {yolo_model_path}")

        self.target_aspect_ratio = 1.4262820512820513
        self.template_size = (312, 445)

        # --- Configuration ---
        self.title_bbox_percent = [0.08, 0.05, 0.65, 0.10]
        self.executor = ThreadPoolExecutor(max_workers=3)
        print("✅ RealTimeCardProcessor Initialization complete.")
        self.last_ocr = ""

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        return rect

    def _detect_cards_with_yolo(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all potential cards using YOLO instance segmentation.
        Returns list of card candidates with their properties.
        """
        # Get original frame dimensions
        original_height, original_width = frame.shape[:2]
        
        # Run YOLO inference
        results = self.yolo_model(frame)
        result = results[0]
        
        if result.masks is None or len(result.masks) == 0:
            return []
            
        card_candidates = []
        
        for i, mask_tensor in enumerate(result.masks.data):
            # Get the corresponding bounding box from YOLO results
            box = result.boxes[i]
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            # Convert mask to numpy and resize to original frame dimensions
            mask = mask_tensor.cpu().numpy()
            mask = cv2.resize(
                mask.astype(float), 
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Find contours in the resized mask
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area < self.MIN_CONTOUR_AREA:
                continue
                
            # Approximate the contour to get a polygon
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
            
            # Accept 4-8 corners for more flexibility
            if 4 <= len(approx) <= self.MAX_CORNERS:
                # Calculate aspect ratio using minimum area rectangle
                _, (w, h), _ = cv2.minAreaRect(approx)
                if w > 0 and h > 0:
                    aspect_ratio = max(w, h) / min(w, h)
                    if abs(aspect_ratio - self.target_aspect_ratio) < self.ASPECT_RATIO_TOLERANCE:
                        # If not 4 points, get the best 4-point approximation
                        if len(approx) != 4:
                            # Use the minimum area rectangle to get 4 points
                            rect = cv2.minAreaRect(approx)
                            box = cv2.boxPoints(rect)
                            approx = np.int32(box).reshape(-1, 1, 2)
                        
                        card_candidates.append({
                            'contour': approx,  # Use actual mask contour (now properly aligned)
                            'mask': mask,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'yolo_bbox': bbox,  # Use YOLO's original bounding box
                            'contour_bbox': cv2.boundingRect(approx)  # Contour-based bounding box
                        })
        
        # Sort by area (largest first)
        card_candidates.sort(key=lambda x: x['area'], reverse=True)
        return card_candidates

    def _warp_card_perspective(self, frame: np.ndarray, card_data: Dict) -> Optional[np.ndarray]:
        """
        Warp a detected card to frontoparallel view.
        """
        try:
            points = card_data['contour']
            
            # Order the points for consistent processing
            ordered_points = self._order_points(points.reshape(-1, 2))
            
            # Correct for perspective
            w_template, h_template = self.template_size
            dst_pts = np.array([[0,0], [w_template-1,0], [w_template-1,h_template-1], [0,h_template-1]], dtype="float32")
            H = cv2.getPerspectiveTransform(ordered_points.astype(np.float32), dst_pts)
            warped_card = cv2.warpPerspective(frame, H, self.template_size)
            
            return warped_card
        except Exception as e:
            print(f"Error in perspective warping: {e}")
            return None

    def _run_ocr_on_card(self, warped_card: np.ndarray) -> str:
        """Runs OCR on the title area of a warped card image."""
        try:
            h, w, _ = warped_card.shape
            x1_pct, y1_pct, x2_pct, y2_pct = self.title_bbox_percent
            tx1, ty1, tx2, ty2 = int(x1_pct * w), int(y1_pct * h), int(x2_pct * w), int(y2_pct * h)
            
            # Ensure valid crop coordinates
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(w, tx2), min(h, ty2)
            
            if tx2 <= tx1 or ty2 <= ty1:
                return ""
                
            title_crop = warped_card[ty1:ty2, tx1:tx2]
            
            if title_crop.size == 0:
                return ""
                
            pil_image = Image.fromarray(cv2.cvtColor(title_crop, cv2.COLOR_BGR2RGB))
            pixel_values = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.ocr_model.generate(pixel_values)
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return "OCR Error"

    def _clean_ocr_text(self, ocr_text: str) -> str:
        """
        Clean OCR text to improve card name recognition.
        """
        if not ocr_text or ocr_text == "OCR Error":
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', ocr_text.strip())
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s\-&\']', '', cleaned)  # Keep only letters, numbers, spaces, hyphens, ampersands, apostrophes
        
        # Remove very short or likely incorrect results
        if len(cleaned) < 3:
            return ""
            
        return cleaned

    def get_card_details(self, card_name: str) -> str:
        """
        Fetches card details from Scryfall API synchronously.
        """
        # Clean the card name first
        clean_card_name = self._clean_ocr_text(card_name)
        if not clean_card_name:
            return "Invalid card name"
        print(clean_card_name)
        # URL encode the cleaned card name
        encoded_card_name = urllib.parse.quote(clean_card_name)
        url = f'https://api.scryfall.com/cards/named?fuzzy={encoded_card_name}'
        headers = {
            'Accept': '*/*',
            'User-Agent': 'WhatsAI MTG Card Processor version 0.0'
        }
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                
                name = data.get('name', 'N/A')
                mana_cost = 'N/A'
                if 'mana_cost' in data:
                    mana_cost = data['mana_cost'].replace('}{', ' ').lstrip('{').rstrip('}')
                type_line = data.get('type_line', 'N/A')
                oracle_text = data.get('oracle_text', 'N/A')
                flavor_text = data.get('flavor_text', 'None')

                power = data.get('power')
                toughness = data.get('toughness')
                pt = f'{power}/{toughness}' if power and toughness else 'N/A'

                details = f'''
Name: {name}
Mana Cost: {mana_cost}
Type: {type_line}
Power/Toughness: {pt}
Oracle Text: {oracle_text}
Flavor Text: {flavor_text}
'''
                return details.strip()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"Card not found: {clean_card_name}"
            else:
                return f"API Error: {e.code}"
        except Exception as e:
            return f"Error fetching card details: {str(e)}"

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        output_frame = frame.copy()
        
        # --- Step 1: Detect all card candidates with YOLO ---
        card_candidates = self._detect_cards_with_yolo(frame)
        
        best_ocr_text = ""
        best_card_data = None
        best_warped_card = None
        
        if card_candidates:
            # --- Step 2: Process the best candidate in parallel ---
            best_candidate = card_candidates[0]
            
            # Start parallel tasks
            future_warp = self.executor.submit(self._warp_card_perspective, frame, best_candidate)
            
            # Wait for warping to complete
            warped_card = future_warp.result()
            
            if warped_card is not None:
                # Start OCR task
                future_ocr = self.executor.submit(self._run_ocr_on_card, warped_card)
                
                # Get OCR result
                ocr_text = future_ocr.result()
                
                if ocr_text and ocr_text != "OCR Error":
                    best_ocr_text = ocr_text
                    best_card_data = best_candidate
                    best_warped_card = warped_card
        
        # --- Step 3: Visualization ---
        if best_card_data is not None:
            # Draw the detected card boundary using the actual mask contour
            cv2.polylines(output_frame, [best_card_data['contour']], True, (0,255,0), 3, cv2.LINE_AA)
            
            # Draw YOLO's original bounding box (red)
            yolo_bbox = best_card_data['yolo_bbox']
            x1, y1, x2, y2 = map(int, yolo_bbox)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0,0,255), 2)
            #cv2.putText(output_frame, "YOLO Box", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            # Draw contour-based bounding box (blue)
            x, y, w, h = best_card_data['contour_bbox']
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255,0,0), 2)
            #cv2.putText(output_frame, "Contour Box", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            
            # Show the warped card if available
            if best_warped_card is not None:
                h_card, w_card, _ = best_warped_card.shape
                x1_pct, y1_pct, x2_pct, y2_pct = self.title_bbox_percent
                tx1, ty1, tx2, ty2 = int(x1_pct*w_card), int(y1_pct*h_card), int(x2_pct*w_card), int(y2_pct*h_card)
                title_crop = best_warped_card[ty1:ty2, tx1:tx2]

                if title_crop.size > 0:
                    h_in, w_in, _ = title_crop.shape
                    w_disp = 250
                    h_disp = int(h_in * (w_disp/w_in)) if w_in > 0 else 0
                    if h_disp > 0:
                        inset = cv2.resize(title_crop, (w_disp, h_disp), interpolation=cv2.INTER_AREA)
                        x_start, y_start = output_frame.shape[1] - w_disp - 10, 10
                        if y_start + h_disp < output_frame.shape[0] and x_start > 0:
                            output_frame[y_start:y_start+h_disp, x_start:x_start+w_disp] = inset
                            cv2.rectangle(output_frame, (x_start, y_start), (x_start+w_disp, y_start+h_disp), (0,0,0), 2)

        # --- Final Text and Status Drawing ---
        if best_ocr_text and best_ocr_text != self.last_ocr: 
            best_ocr_text = self.get_card_details(best_ocr_text)
            self.last_ocr = best_ocr_text
        
        # Status shows detection info
        status = f"CARD DETECTED ({len(card_candidates)} candidates)" if card_candidates else "NO CARD"
        print(f"Status: {status}")
        
        return output_frame, best_ocr_text

    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        return None, {"message": "RealTimeCardProcessor does not process point cloud data."}

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

processor = RealTimeCardProcessor('microsoft/trocr-base-printed', "./models/yolo11n-seg.pt")
app = processor.app