import json
import numpy as np
import cv2
from PIL import Image
import torch
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys
import open3d as o3d

# Add llava parent directory to sys.path
llava_parent_dir = r'/home/znasif/vision-depth-pro'
if llava_parent_dir not in sys.path:
    sys.path.insert(0, llava_parent_dir)

from depth_pro import create_model_and_transforms, load_rgb

app = FastAPI()

class DepthProcessor:
    def __init__(self, 
                 checkpoint_uri="/home/znasif/vision-depth-pro/checkpoints/depth_pro.pt",
                 use_gpu=True,
                 voxel_size: float = 0.01,
                 estimate_normals: bool = True):
        self.checkpoint_uri = checkpoint_uri
        self.voxel_size = voxel_size
        self.estimate_normals = estimate_normals
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self._load_model()

    def _load_model(self):
        self.model, self.transform = create_model_and_transforms(
            device=self.device,
            precision=torch.half
        )
        self.model.eval()
        print(f"DepthPro model loaded successfully on {self.device}")

    def preprocess_image(self, frame: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return pil_image, frame_rgb

    def pointcloud_to_json(self, pcd: o3d.geometry.PointCloud) -> dict:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        return {
            'points': points.tolist(),
            'colors': colors.tolist() if colors is not None else [],
            'normals': normals.tolist() if normals is not None else []
        }

    def create_point_cloud(self, depth: np.ndarray, rgb_image: np.ndarray, focallength_px: torch.Tensor) -> o3d.geometry.PointCloud:
        height, width = depth.shape
        cx, cy = width / 2, height / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=focallength_px,
            fy=focallength_px,
            cx=cx,
            cy=cy
        )
        focallength_px = focallength_px.detach().cpu().numpy().item()
        y, x = np.indices((height, width))
        z = depth
        z[z == 0] = 1e-6
        X = (x - cx) * z / focallength_px
        Y = (y - cy) * z / focallength_px
        Z = z
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        colors = rgb_image.reshape(-1, 3) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        if self.estimate_normals:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        return pcd

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        output = frame.copy()
        pil_image, rgb_image = self.preprocess_image(frame)
        try:
            image_tensor = self.transform(pil_image)
            with torch.no_grad():
                prediction = self.model.infer(image_tensor)
            depth = prediction["depth"]
            focallength_px = prediction["focallength_px"]
            depth_np = depth.detach().cpu().numpy().squeeze()
            pcd = self.create_point_cloud(depth_np, rgb_image, focallength_px)
            pointcloud_json = self.pointcloud_to_json(pcd)
            return output, pointcloud_json
        except Exception as e:
            print(f"Processing error: {e}")
            return output, {'error': str(e), 'points': [], 'colors': [], 'normals': []}

# Initialize processor
processor = DepthProcessor()

class ProcessRequest(BaseModel):
    image: str

@app.post("/process")
async def process(request: ProcessRequest):
    try:
        # Extract base64 data
        if request.image.startswith('data:image/jpeg;base64,'):
            encoded_data = request.image.split('base64,')[1]
        else:
            encoded_data = request.image

        # Decode base64 data
        decoded_data = base64.b64decode(encoded_data)
        nparr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode input frame")

        # Process frame
        processed_frame, result = processor.process_frame(frame)

        # Encode processed frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        success, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
        if not success:
            raise ValueError("Failed to encode processed frame")

        processed_data = base64.b64encode(buffer).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{processed_data}"

        return {"image": image_data, "text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))