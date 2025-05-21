import json
import sys
import numpy as np
import cv2
from PIL import Image
import torch
from pathlib import Path
import open3d as o3d
from typing import Tuple, Dict, Any, List
from scipy.spatial.transform import Rotation # For SceneScript output processing

# --- Path Setup ---
# Path to the directory containing the 'depth_pro' package
llava_parent_dir = r'/home/znasif/vision-depth-pro'
if llava_parent_dir not in sys.path:
    sys.path.insert(0, llava_parent_dir)

# IMPORTANT: Path to the root of the 'scenescript' repository
# This directory should contain the 'src' folder of the scenescript library
scenescript_root_dir = r'/home/znasif/scenescript'
if scenescript_root_dir not in sys.path:
    sys.path.insert(0, scenescript_root_dir)

# --- Imports ---
try:
    from depth_pro import create_model_and_transforms, load_rgb
except ImportError as e:
    print(f"Error importing depth_pro: {e}. Ensure '{llava_parent_dir}' is correct and contains 'depth_pro'.")
    sys.exit(1)

try:
    from src.networks.scenescript_model import SceneScriptWrapper
    # from src.data.language_sequence import LanguageSequence # Might not be needed directly
    # from src.data.point_cloud import PointCloud # For type hinting or if SceneScript requires this object
except ImportError as e:
    print(f"Error importing from scenescript 'src': {e}. "
          f"Ensure '{scenescript_root_dir}' is correct and added to sys.path, "
          f"and the SceneScript library is correctly structured there.")
    sys.exit(1)


# --- BaseProcessor Definition (if not in a separate file) ---
class BaseProcessor:
    def __init__(self):
        pass
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError
# --- End BaseProcessor Definition ---


class SceneScriptProcessor(BaseProcessor):
    def __init__(self,
                 depthpro_checkpoint_uri="/home/znasif/vision-depth-pro/checkpoints/depth_pro.pt",
                 scenescript_checkpoint_uri: str = "/home/znasif/vidServer/server/models/scenescript_model_non_manhattan_class_agnostic_model.ckpt",
                 use_gpu=True,
                 voxel_size: float = 0.01,
                 estimate_normals: bool = True,
                 scenescript_nucleus_sampling_thresh: float = 0.05):
        """
        Initialize DepthPro and SceneScript processor.

        Args:
            depthpro_checkpoint_uri (str): Path to the DepthPro model checkpoint.
            scenescript_checkpoint_uri (str): Path to the SceneScript model checkpoint (.ckpt).
            use_gpu (bool): Whether to use GPU acceleration.
            voxel_size (float): Voxel size for point cloud downsampling.
            estimate_normals (bool): Whether to estimate surface normals.
            scenescript_nucleus_sampling_thresh (float): Nucleus sampling threshold for SceneScript.
        """
        super().__init__()
        self.depthpro_checkpoint_uri = depthpro_checkpoint_uri
        self.scenescript_checkpoint_uri = scenescript_checkpoint_uri
        self.voxel_size = voxel_size
        self.estimate_normals = estimate_normals
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.scenescript_nucleus_sampling_thresh = scenescript_nucleus_sampling_thresh

        self._load_depthpro_model()
        self._load_scenescript_model()

    def _load_depthpro_model(self):
        self.depthpro_model, self.depthpro_transform = create_model_and_transforms(
            device=self.device,
            precision=torch.half
        )
        # Assuming create_model_and_transforms loads the checkpoint or it's part of the model init.
        # If explicit loading is needed for DepthPro:
        # try:
        #     checkpoint = torch.load(self.depthpro_checkpoint_uri, map_location=self.device)
        #     # Adjust state_dict key if necessary based on how depth_pro.pt is saved
        #     # self.depthpro_model.load_state_dict(checkpoint['model_state_dict'])
        #     print(f"DepthPro checkpoint '{self.depthpro_checkpoint_uri}' loaded.")
        # except Exception as e:
        #     print(f"Warning: Could not load DepthPro checkpoint from {self.depthpro_checkpoint_uri}: {e}")
        #     print("DepthPro model will use its initial weights (if any) or fail if checkpoint is mandatory.")

        self.depthpro_model.eval()
        print(f"DepthPro model loaded/configured successfully on {self.device}.")


    def _load_scenescript_model(self):
        print(f"Attempting to load SceneScript model from checkpoint: {self.scenescript_checkpoint_uri}")
        try:
            self.scenescript_model = SceneScriptWrapper.load_from_checkpoint(
                self.scenescript_checkpoint_uri
            ).to(self.device)
            self.scenescript_model.eval()
            print(f"SceneScript model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading SceneScript model: {e}")
            print("Please ensure the checkpoint path is correct and the SceneScript environment is set up.")
            self.scenescript_model = None


    def preprocess_image(self, frame: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return pil_image, frame_rgb

    def pointcloud_to_json(self, pcd: o3d.geometry.PointCloud) -> Dict:
        points = np.asarray(pcd.points).tolist()
        colors = np.asarray(pcd.colors).tolist() if pcd.has_colors() else []
        normals = np.asarray(pcd.normals).tolist() if pcd.has_normals() else []
        return {'points': points, 'colors': colors, 'normals': normals}

    def create_point_cloud(self, depth: np.ndarray, rgb_image: np.ndarray, focallength_px_tensor: torch.Tensor) -> o3d.geometry.PointCloud:
        height, width = depth.shape
        focallength_px = focallength_px_tensor.detach().cpu().item()
        cx, cy = width / 2.0, height / 2.0

        color_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, focallength_px, focallength_px, cx, cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        if self.estimate_normals and len(pcd.points) > 0:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=30))
            pcd.orient_normals_towards_camera_location()
        return pcd

    def _parse_scenescript_entities(self, lang_seq_entities: List[Any]) -> List[Dict]:
        """
        Parses entities from SceneScript's LanguageSequence output into a structured list.
        Focuses on 'make_bbox' and other relevant geometric entities.
        """
        parsed_objects = []
        for entity in lang_seq_entities:
            obj_data = {
                "id": entity.params.get("id"),
                "command": entity.COMMAND_STRING,
                "params_raw": entity.params # Store raw params for debugging or other uses
            }

            if entity.COMMAND_STRING == "make_bbox":
                obj_data["class_name"] = entity.params.get("class", "unknown_bbox")
                obj_data["position"] = [
                    entity.params.get("position_x", 0),
                    entity.params.get("position_y", 0),
                    entity.params.get("position_z", 0)
                ]
                obj_data["scale"] = [
                    entity.params.get("scale_x", 0),
                    entity.params.get("scale_y", 0),
                    entity.params.get("scale_z", 0)
                ]
                angle_z = entity.params.get("angle_z", 0)
                # Convert rotation to a more common format (e.g., quaternion or axis-angle list)
                # SceneScript notebook uses scipy.Rotation
                rotation_matrix = Rotation.from_rotvec([0, 0, angle_z]).as_matrix()
                obj_data["rotation_matrix"] = rotation_matrix.tolist() # For JSON
                obj_data["angle_z"] = angle_z

            elif entity.COMMAND_STRING == "make_wall":
                obj_data["class_name"] = "wall"
                obj_data["height"] = entity.params.get("height")
                obj_data["thickness"] = entity.params.get("thickness", 0.0) # Often 0 in example
                corner_a = np.array([entity.params.get("a_x"), entity.params.get("a_y"), entity.params.get("a_z")])
                corner_b = np.array([entity.params.get("b_x"), entity.params.get("b_y"), entity.params.get("b_z")])
                length = np.linalg.norm(corner_a - corner_b)
                direction = corner_b - corner_a
                angle = np.arctan2(direction[1], direction[0]) if np.linalg.norm(direction[:2]) > 1e-6 else 0.0
                centre = (corner_a + corner_b) * 0.5 + np.array([0, 0, 0.5 * obj_data["height"]])
                
                obj_data["position"] = centre.tolist()
                obj_data["scale"] = [length, obj_data["thickness"], obj_data["height"]] # length, thickness, height
                obj_data["angle_z"] = angle # Rotation around Z
                rotation_matrix = Rotation.from_rotvec([0, 0, angle]).as_matrix()
                obj_data["rotation_matrix"] = rotation_matrix.tolist()


            # Add parsers for "make_door", "make_window" if needed, similar to `language_to_bboxes`
            # For now, we only fully parse bbox and wall. Others will have raw params.
            
            parsed_objects.append(obj_data)
        return parsed_objects

    def run_scenescript_inference(self, pcd: o3d.geometry.PointCloud) -> Dict:
        if self.scenescript_model is None:
            return {"error": "SceneScript model not loaded", "objects": []}
        if not pcd.has_points():
            return {"error": "Empty point cloud for SceneScript", "objects": []}

        points_np = np.asarray(pcd.points)
        
        # SceneScript might also use colors or normals if its PointCloud object or model supports it.
        # For now, using only points as shown in the inference notebook: `point_cloud_obj.points`
        # colors_np = np.asarray(pcd.colors) if pcd.has_colors() else None
        # point_cloud_for_scenescript = point_cloud_obj_from_numpy(points_np, colors_np) # If SceneScript uses a specific PointCloud object

        print(f"Running SceneScript inference with {len(points_np)} points.")
        try:
            with torch.no_grad():
                lang_seq = self.scenescript_model.run_inference(
                    points_np, # Pass points as NumPy array
                    nucleus_sampling_thresh=self.scenescript_nucleus_sampling_thresh,
                    # verbose=True # Optional
                )
            
            # lang_seq is a LanguageSequence object, its entities contain the scene description
            parsed_objects = self._parse_scenescript_entities(lang_seq.entities)
            return {"objects": parsed_objects}

        except Exception as e:
            print(f"SceneScript inference error: {e}")
            return {"error": str(e), "objects": []}

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        output_frame = frame.copy()
        pil_image, rgb_image_np = self.preprocess_image(frame)

        results = {
            'point_cloud': {'error': 'DepthPro processing failed', 'points': [], 'colors': [], 'normals': []},
            'scene_understanding': {'error': 'SceneScript processing not run or failed', 'objects': []}
        }

        try:
            # --- DepthPro Inference ---
            image_tensor = self.depthpro_transform(pil_image).to(self.device)
            if image_tensor.ndim == 3: image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():
                prediction = self.depthpro_model.infer(image_tensor)
            depth_tensor, focallength_px_tensor = prediction["depth"], prediction["focallength_px"]
            depth_np = depth_tensor.squeeze().cpu().numpy()

            # --- Point Cloud Generation ---
            pcd = self.create_point_cloud(depth_np, rgb_image_np, focallength_px_tensor)
            results['point_cloud'] = self.pointcloud_to_json(pcd)

            # --- SceneScript Inference ---
            if self.scenescript_model and pcd.has_points():
                scene_output = self.run_scenescript_inference(pcd)
                results['scene_understanding'] = scene_output
            elif not self.scenescript_model:
                 results['scene_understanding']['error'] = "SceneScript model not loaded."
            elif not pcd.has_points():
                 results['scene_understanding']['error'] = "Skipped SceneScript due to empty point cloud."


        except Exception as e:
            print(f"Main processing error: {e}")
            # Ensure error is logged in the appropriate part of results
            if not results['point_cloud'].get('points'): # If point cloud failed
                 results['point_cloud']['error'] = str(e)
            results['scene_understanding']['error'] = f"Processing error before/during SceneScript: {e}"
            
        return output_frame, results


if __name__ == '__main__':
    print("Starting SceneScriptProcessor Test Script")

    # --- IMPORTANT: User Configuration ---
    # 1. Update this path to the root of your cloned SceneScript repository
    #    This is the directory that contains the 'src' folder for SceneScript.
    #    Example: '/home/your_user/projects/scenescript'
    scenescript_repo_root = "/path/to/your/scenescript_repository" # <--- MUST BE SET BY USER

    # 2. Update this path to your downloaded SceneScript model checkpoint (.ckpt file)
    #    Example: '/home/your_user/models/scenescript/checkpoint.ckpt'
    scenescript_model_checkpoint = "/path/to/your/scenescript_model.ckpt" # <--- MUST BE SET BY USER
    
    # 3. (Optional) Update path to your DepthPro checkpoint if not default
    depthpro_model_checkpoint = "/home/znasif/vision-depth-pro/checkpoints/depth_pro.pt"


    # --- Automated Path Setup ---
    if scenescript_repo_root == "/path/to/your/scenescript_repository":
        print("ERROR: 'scenescript_repo_root' is not set. Please update it in the __main__ block.")
        sys.exit(1)
    if scenescript_repo_root not in sys.path:
        sys.path.insert(0, scenescript_repo_root)
        print(f"Added '{scenescript_repo_root}' to sys.path for SceneScript imports.")

    # Re-check SceneScript imports after path modification, just in case.
    try:
        from src.networks.scenescript_model import SceneScriptWrapper
    except ImportError:
        print(f"FATAL: Could not import SceneScriptWrapper even after adding '{scenescript_repo_root}' to sys.path.")
        print("Please ensure the path is correct and the 'src' directory exists there.")
        sys.exit(1)
    
    if scenescript_model_checkpoint == "/path/to/your/scenescript_model.ckpt":
        print("WARNING: 'scenescript_model_checkpoint' is using a placeholder path.")
        print("SceneScript model will likely fail to load. Update this path.")
        # Allow to continue for testing other parts, but SceneScript won't work.


    # --- Initialize Processor ---
    print("Initializing SceneScriptProcessor...")
    try:
        processor = SceneScriptProcessor(
            depthpro_checkpoint_uri=depthpro_model_checkpoint,
            scenescript_checkpoint_uri=scenescript_model_checkpoint,
            use_gpu=torch.cuda.is_available(),
            voxel_size=0.02,
            estimate_normals=True
        )
    except Exception as e:
        print(f"Error initializing SceneScriptProcessor: {e}")
        sys.exit(1)
    print("SceneScriptProcessor initialized.")

    # --- Load or Create a Test Image ---
    frame = None
    try:
        # To use a real image:
        # test_image_path = "/path/to/your/image.jpg"
        # frame = cv2.imread(test_image_path)
        # if frame is None:
        #     raise FileNotFoundError(f"Image not found at {test_image_path}")
        # else:
        #     print(f"Loaded test image from {test_image_path}")
        
        # For now, using a dummy image:
        print("Creating a dummy OpenCV frame for testing (640x480).")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    except Exception as e:
        print(f"Error loading/creating test image: {e}")
        sys.exit(1)

    # --- Process the Frame ---
    if frame is not None:
        print(f"Processing frame with shape: {frame.shape}...")
        original_frame, results = processor.process_frame(frame)
        print("\n--- Processing Complete ---")

        print("\nPoint Cloud Data Summary:")
        if results['point_cloud'].get('points') and len(results['point_cloud']['points']) > 0:
            print(f"  Number of points: {len(results['point_cloud']['points'])}")
            print(f"  Number of colors: {len(results['point_cloud']['colors'])}")
            print(f"  Number of normals: {len(results['point_cloud']['normals'])}")
            print(f"  First 3 points: {np.array(results['point_cloud']['points'])[:3].tolist()}")
        else:
            print(f"  Error: {results['point_cloud'].get('error', 'No point cloud data')}")

        print("\nScene Understanding Data:")
        if results['scene_understanding'].get('objects'):
            print(f"  Number of detected objects/entities: {len(results['scene_understanding']['objects'])}")
            for i, obj in enumerate(results['scene_understanding']['objects']):
                print(f"  Object {i+1}:")
                print(f"    ID: {obj.get('id')}")
                print(f"    Command: {obj.get('command')}")
                print(f"    Class Name: {obj.get('class_name', 'N/A')}")
                print(f"    Position: {obj.get('position', 'N/A')}")
                # print(f"    Raw Params: {obj.get('params_raw')}") # Uncomment for full details
                if i >= 2: # Print details for first 3 objects only
                    print("    (More objects exist, details truncated for brevity)")
                    break
        else:
            print(f"  Error: {results['scene_understanding'].get('error', 'No scene understanding data or objects detected')}")
        
        # --- Optionally save the results ---
        # output_json_path = "processed_scene_data.json"
        # with open(output_json_path, "w") as f:
        #     json.dump(results, f, indent=2)
        # print(f"\nFull results saved to {output_json_path}")
    else:
        print("Failed to load or create a test frame. Exiting.")