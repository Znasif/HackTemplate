import torch
import torchvision.transforms as TTR
import numpy as np
import cv2
import math
import json
import os
import tempfile
import subprocess
from PIL import Image
import open3d as o3d
from typing import Tuple, Dict, Optional, Union, List
import threading
import time
from .base_processor import BaseProcessor
from depth_pro import create_model_and_transforms

class FDSConfig:
    """Configuration class for FDS simulation parameters."""
    
    def __init__(self):
        self.material_properties = {
            'A1': {
                'density': 2300,
                'conductivity': 1.6,
                'specific_heat': 1000,
                'emissivity': 0.9,
                'ignition_temperature': None,
                'heat_of_combustion': 0,
                'soot_yield': 0,
                'co_yield': 0,
                'color': 'GRAY',
                'burn_duration': float('inf')  # Non-combustible
            },
            'A2': {
                'density': 600,
                'conductivity': 0.15,
                'specific_heat': 1500,
                'emissivity': 0.85,
                'ignition_temperature': 300,
                'heat_of_combustion': 10,
                'soot_yield': 0.01,
                'co_yield': 0.004,
                'color': 'BROWN',
                'burn_duration': 30.0  # Seconds
            },
            'B/C': {
                'density': 500,
                'conductivity': 0.12,
                'specific_heat': 1600,
                'emissivity': 0.8,
                'ignition_temperature': 250,
                'heat_of_combustion': 18,
                'soot_yield': 0.02,
                'co_yield': 0.008,
                'color': 'ORANGE',
                'burn_duration': 20.0
            },
            'D/E': {
                'density': 400,
                'conductivity': 0.08,
                'specific_heat': 1400,
                'emissivity': 0.85,
                'ignition_temperature': 200,
                'heat_of_combustion': 25,
                'soot_yield': 0.15,
                'co_yield': 0.05,
                'color': 'RED',
                'burn_duration': 15.0
            },
            'F': {
                'density': 300,
                'conductivity': 0.05,
                'specific_heat': 1200,
                'emissivity': 0.9,
                'ignition_temperature': 150,
                'heat_of_combustion': 35,
                'soot_yield': 0.25,
                'co_yield': 0.08,
                'color': 'MAGENTA',
                'burn_duration': 10.0
            }
        }
        self.simulation_time = 60.0
        self.output_interval = 1.0
        self.grid_resolution = 0.1
        self.ambient_temperature = 20.0
        
    def get_hrr_per_unit_area(self, flammability_class: str) -> float:
        hrr_map = {
            'A1': 0,
            'A2': 50,
            'B/C': 200,
            'D/E': 500,
            'F': 1000
        }
        return hrr_map.get(flammability_class, 0)

class FlameProcessor(BaseProcessor):
    def __init__(self, 
                 jit_path: str = None, 
                 taxonomy_path: str = None,
                 depth_checkpoint_uri: str = None,
                 use_gpu: bool = True,
                 voxel_size: float = 0.01,
                 estimate_normals: bool = True,
                 fds_executable: str = None,
                 enable_fds: bool = True,
                 time_step: float = 1.0):
        super().__init__()
        self.base_pointcloud = None
        self.base_material_labels = None
        self.fire_simulation_active = False
        self.current_simulation_time = 0.0
        self.time_step = time_step
        self.ignition_point = None
        self.fire_history = []
        self.base_frame_processed = False
        self.fire_states = {}
        self.simulation_thread = None
        self.state_lock = threading.Lock()
        self.max_simulation_time = 60.0
        self.fds_decay_data = {}  # Store FDS decay thresholds
        self.fds_config = FDSConfig()
        self.enable_fds = enable_fds
        self.forced_ignition_index: Optional[int] = None
        self.fds_executable = fds_executable or os.environ.get('FDS_EXECUTABLE') or self._find_fds_executable()
        if enable_fds and not self.fds_executable:
            print("WARNING: FDS executable not found. Disabling FDS simulation.")
            self.enable_fds = False
        
        if jit_path is None:
            jit_path = '/home/znasif/vidServer/server/models/DMS46_v1.pt'
        if taxonomy_path is None:
            taxonomy_path = '/home/znasif/vidServer/server/models/taxonomy.json'
        self.vis_bgr = None
        try:
            with open(taxonomy_path, 'rb') as f:
                t = json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load taxonomy: {e}")
            raise

        self.dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                      23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                      46, 47, 48, 49, 50, 51, 52, 53, 56]
        self.material_names = [t['names'][i] for i in self.dms46]
        original_colormap = np.array([t['srgb_colormap'][i] for i in self.dms46], dtype=np.uint8)
        self.flammability_map = {
            1: 'A1', 2: 'B/C', 3: 'A1', 4: 'D/E', 5: 'D/E', 6: 'B/C', 7: 'A1', 8: 'B/C',
            9: 'D/E', 10: 'A1', 11: 'B/C', 12: 'A1', 13: 'D/E', 15: 'F', 16: 'D/E', 17: 'D/E',
            18: 'D/E', 19: 'D/E', 20: 'A2', 21: 'D/E', 23: 'D/E', 24: 'D/E', 26: 'A1',
            27: 'A2', 29: 'B/C', 30: 'D/E', 32: 'D/E', 33: 'D/E', 34: 'D/E', 35: 'D/E',
            36: 'D/E', 37: 'D/E', 38: 'D/E', 39: 'D/E', 41: 'D/E', 43: 'A1', 44: 'A1',
            46: 'A1', 47: 'B/C', 48: 'D/E', 49: 'D/E', 50: 'B/C', 51: 'D/E', 52: 'D/E',
            53: 'D/E', 56: 'D/E'
        }
        self.class_colors = {
            'A1': original_colormap[self.dms46.index(26)],
            'A2': original_colormap[self.dms46.index(20)],
            'B/C': original_colormap[self.dms46.index(29)],
            'D/E': original_colormap[self.dms46.index(52)],
            'F': original_colormap[self.dms46.index(15)]
        }
        self.srgb_colormap = np.array([self.class_colors[self.flammability_map[i]] for i in self.dms46], dtype=np.uint8)
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        try:
            self.material_model = torch.jit.load(jit_path, map_location=self.device)
            self.material_model.eval()
        except Exception as e:
            print(f"ERROR: Failed to load material model: {e}")
            raise
        if depth_checkpoint_uri is None:
            depth_checkpoint_uri = '/home/znasif/vision-depth-pro/checkpoints/depth_pro.pt'
        self.depth_checkpoint_uri = depth_checkpoint_uri
        self.voxel_size = voxel_size
        self.estimate_normals = estimate_normals
        self._load_depth_model()

    def _find_fds_executable(self) -> Optional[str]:
        import shutil
        fds_names = ['fds', 'fds.exe', 'fds6']
        for name in fds_names:
            path = shutil.which(name)
            if path:
                print(f"DEBUG: Found FDS executable at: {path}")
                return path
        return None

    def _load_depth_model(self):
        self.depth_model, self.depth_transform = create_model_and_transforms(
            device=self.device,
            precision=torch.half if self.device == 'cuda' else torch.float32
        )
        self.depth_model.eval()

    def preprocess_image(self, frame: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported number of channels: {frame.shape[2]}")
        pil_image = Image.fromarray(frame_rgb)
        return pil_image, frame_rgb

    def pointcloud_to_json(self, pcd: o3d.geometry.PointCloud, material_labels: Optional[np.ndarray] = None, 
                           intensities: Optional[Union[np.ndarray, List]] = None, 
                           temperatures: Optional[Union[np.ndarray, List]] = None, 
                           burned_out: Optional[Union[np.ndarray, List]] = None) -> Dict:
        """Convert an Open3D point cloud to a JSON-serializable dictionary with fire effects in colors."""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros((len(points), 3))
        normals = np.asarray(pcd.normals) if pcd.has_normals() else np.array([])
        
        # Encode fire effects in colors
        if intensities is not None and burned_out is not None:
            # Ensure inputs are numpy arrays for calculations
            intensities = np.asarray(intensities)
            burned_out = np.asarray(burned_out)

            colors = colors.copy()  # Avoid modifying original
            burning_mask = intensities > 0

            # --- START OF FIX ---
            # Only attempt to apply colors if there are actually points burning.
            if np.any(burning_mask):
                burning_intensities = intensities[burning_mask]
                # Create the color array for only the burning points
                new_colors = np.array([[1.0, 0.5 - 0.5 * intensity, 0.0] for intensity in burning_intensities])
                # Assign the new colors using the boolean mask
                colors[burning_mask] = new_colors
            
            if np.any(burned_out):
                colors[burned_out] = [0.0, 0.0, 0.0]  # Black for burned-out points
            # --- END OF FIX ---
        
        result = {
            'points': points.tolist(),
            'colors': colors.tolist(),
            'normals': normals.tolist()
        }
        
        if material_labels is not None:
            result['material_labels'] = material_labels.tolist()
        if intensities is not None:
            # np.asarray is used to handle both original list and converted numpy array
            result['intensities'] = np.asarray(intensities).tolist()
        if temperatures is not None:
            result['temperatures'] = np.asarray(temperatures).tolist()
        if burned_out is not None:
            result['burned_out'] = np.asarray(burned_out).tolist()
            
        return result

    # In your flame_processor.py, replace the existing method with this one.

    def create_point_cloud_with_materials(self, depth: np.ndarray, original_vis_bgr: np.ndarray, 
                                        material_mask: np.ndarray, focallength_px_tensor: torch.Tensor) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Creates a point cloud with natural RGB colors and correctly mapped material labels,
        even after voxel downsampling.
        """
        height, width = depth.shape; cx, cy = width / 2.0, height / 2.0
        focallength_px = focallength_px_tensor.item()
        y_indices, x_indices = np.indices((height, width), dtype=np.float32)
        z = depth.astype(np.float32); z[z < 1e-6] = 1e-6
        X = (x_indices - cx) * z / focallength_px; Y = (y_indices - cy) * z / focallength_px
        points = np.stack((X, Y, z), axis=-1).reshape(-1, 3)
        
        vis_rgb = cv2.cvtColor(original_vis_bgr, cv2.COLOR_BGR2RGB)
        colors_normalized = vis_rgb.reshape(-1, 3) / 255.0
        
        material_labels = material_mask.reshape(-1)
        valid_mask = (material_labels >= 0) & (material_labels < len(self.dms46)); material_labels[~valid_mask] = 0
        
        max_points = 1_000_000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points, colors_normalized, material_labels = points[indices], colors_normalized[indices], material_labels[indices]
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        if self.voxel_size > 0 and len(pcd.points) > 0:
            pcd_down = pcd.voxel_down_sample(self.voxel_size)

            if len(pcd_down.points) > 0:
                kdtree = o3d.geometry.KDTreeFlann(pcd)
                
                down_sampled_labels = []
                for point in pcd_down.points:
                    [k, idx, _] = kdtree.search_knn_vector_3d(point, 1)
                    if k > 0:
                        down_sampled_labels.append(material_labels[idx[0]])
                
                if down_sampled_labels:
                    material_labels = np.array(down_sampled_labels, dtype=np.int64)
                    pcd = pcd_down
                else:
                    pcd = o3d.geometry.PointCloud()
                    material_labels = np.array([])
            else:
                pcd = o3d.geometry.PointCloud()
                material_labels = np.array([])

        if self.estimate_normals and len(pcd.points) >= 30:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=30))
            
        return pcd, material_labels

    def create_point_cloud_with_materials1(self, depth: np.ndarray, original_vis_bgr: np.ndarray, 
                                        material_mask: np.ndarray, focallength_px_tensor: torch.Tensor) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Creates a point cloud with natural RGB colors and correctly mapped material labels,
        even after voxel downsampling.
        """
        height, width = depth.shape; cx, cy = width / 2.0, height / 2.0
        focallength_px = focallength_px_tensor.item()
        y_indices, x_indices = np.indices((height, width), dtype=np.float32)
        z = depth.astype(np.float32); z[z < 1e-6] = 1e-6
        X = (x_indices - cx) * z / focallength_px; Y = (y_indices - cy) * z / focallength_px
        points = np.stack((X, Y, z), axis=-1).reshape(-1, 3)
        
        vis_rgb = cv2.cvtColor(original_vis_bgr, cv2.COLOR_BGR2RGB)
        colors_normalized = vis_rgb.reshape(-1, 3) / 255.0
        
        material_labels = material_mask.reshape(-1)
        valid_mask = (material_labels >= 0) & (material_labels < len(self.dms46)); material_labels[~valid_mask] = 0
        
        max_points = 1_000_000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points, colors_normalized, material_labels = points[indices], colors_normalized[indices], material_labels[indices]
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        if self.voxel_size > 0 and len(pcd.points) > 0:
            # 1. Downsample the point cloud.
            pcd_down = pcd.voxel_down_sample(self.voxel_size)

            if len(pcd_down.points) > 0:
                # 2. Build a KDTree on the ORIGINAL dense point cloud for fast searching.
                kdtree = o3d.geometry.KDTreeFlann(pcd)
                
                # 3. For each point in the NEW downsampled cloud, find its nearest neighbor in the OLD cloud.
                down_sampled_labels = []
                for point in pcd_down.points:
                    [k, idx, _] = kdtree.search_knn_vector_3d(point, 1)
                    if k > 0:
                        down_sampled_labels.append(material_labels[idx[0]])
                
                # Make sure the list is not empty before creating numpy array
                if down_sampled_labels:
                    material_labels = np.array(down_sampled_labels, dtype=np.int64)
                    pcd = pcd_down
                else: # All points were filtered out
                    pcd = o3d.geometry.PointCloud()
                    material_labels = np.array([])

            else:
                # If downsampling resulted in an empty cloud, clear everything.
                pcd = o3d.geometry.PointCloud()
                material_labels = np.array([])

        if self.estimate_normals and len(pcd.points) >= 30:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=30))
            
        return pcd, material_labels

    def create_point_cloud(self, depth: np.ndarray, vis_bgr: np.ndarray, focallength_px_tensor: torch.Tensor) -> o3d.geometry.PointCloud:
        pcd, _ = self.create_point_cloud_with_materials(depth, vis_bgr, np.zeros_like(depth), focallength_px_tensor)
        return pcd

    def apply_color(self, label_mask: np.ndarray) -> np.ndarray:
        vis = np.take(self.srgb_colormap, label_mask, axis=0)
        return vis[..., ::-1]

    def pointcloud_to_fds_geometry(self, pcd: o3d.geometry.PointCloud, material_labels: np.ndarray) -> str:
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return ""
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        geometry_blocks = []
        material_definitions = []
        used_materials = set()
        for label in np.unique(material_labels):
            if 0 <= label < len(self.dms46):
                material_id = self.dms46[label]
                if material_id in self.flammability_map:
                    flam_class = self.flammability_map[material_id]
                    if flam_class not in used_materials:
                        used_materials.add(flam_class)
                        props = self.fds_config.material_properties[flam_class]
                        mat_def = f"""&MATL ID='{flam_class}'
      DENSITY={props['density']}
      CONDUCTIVITY={props['conductivity']}
      SPECIFIC_HEAT={props['specific_heat']}
      EMISSIVITY={props['emissivity']}"""
                        if props['ignition_temperature'] is not None:
                            mat_def += f"\n      IGNITION_TEMPERATURE={props['ignition_temperature']}"
                            mat_def += f"\n      HEAT_OF_COMBUSTION={props['heat_of_combustion']*1000}"
                            mat_def += f"\n      SOOT_YIELD={props['soot_yield']}"
                            mat_def += f"\n      CO_YIELD={props['co_yield']}"
                        mat_def += " /\n"
                        material_definitions.append(mat_def)
                        surf_def = f"&SURF ID='{flam_class}_SURF' MATL_ID='{flam_class}' THICKNESS=0.02"
                        if props['ignition_temperature'] is not None:
                            hrr = self.fds_config.get_hrr_per_unit_area(flam_class)
                            if hrr > 0:
                                surf_def += f" HRRPUA={hrr}"
                        surf_def += f" COLOR='{props['color']}' /\n"
                        material_definitions.append(surf_def)
        grid_size = self.fds_config.grid_resolution
        x_bins = np.arange(min_bound[0], max_bound[0] + grid_size, grid_size)
        y_bins = np.arange(min_bound[1], max_bound[1] + grid_size, grid_size)
        z_bins = np.arange(min_bound[2], max_bound[2] + grid_size, grid_size)
        x_indices = np.digitize(points[:, 0], x_bins) - 1
        y_indices = np.digitize(points[:, 1], y_bins) - 1
        z_indices = np.digitize(points[:, 2], z_bins) - 1
        for material_class in used_materials:
            material_points = []
            for i, label in enumerate(material_labels):
                if 0 <= label < len(self.dms46):
                    material_id = self.dms46[label]
                    if material_id in self.flammability_map and self.flammability_map[material_id] == material_class:
                        material_points.append(i)
            if not material_points:
                continue
            mat_x_indices = x_indices[material_points]
            mat_y_indices = y_indices[material_points]
            mat_z_indices = z_indices[material_points]
            unique_cells = set(zip(mat_x_indices, mat_y_indices, mat_z_indices))
            for x_idx, y_idx, z_idx in unique_cells:
                if (0 <= x_idx < len(x_bins)-1 and 
                    0 <= y_idx < len(y_bins)-1 and 
                    0 <= z_idx < len(z_bins)-1):
                    x1, x2 = x_bins[x_idx], x_bins[x_idx + 1]
                    y1, y2 = y_bins[y_idx], y_bins[y_idx + 1]
                    z1, z2 = z_bins[z_idx], z_bins[z_idx + 1]
                    obst_def = f"&OBST XB={x1:.3f},{x2:.3f},{y1:.3f},{y2:.3f},{z1:.3f},{z2:.3f} SURF_ID='{material_class}_SURF' /\n"
                    geometry_blocks.append(obst_def)
        return "\n".join(material_definitions) + "\n" + "".join(geometry_blocks)

    def create_fds_input_file(self, pcd: o3d.geometry.PointCloud, material_labels: np.ndarray, 
                            ignition_point: Tuple[float, float, float] = None) -> str:
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        domain_padding = 0.5
        x1, y1, z1 = min_bound - domain_padding
        x2, y2, z2 = max_bound + domain_padding
        if x2 - x1 < 2.0: x2 = x1 + 2.0
        if y2 - y1 < 2.0: y2 = y1 + 2.0
        if z2 - z1 < 2.0: z2 = z1 + 2.0
        nx = max(20, int((x2 - x1) / self.fds_config.grid_resolution))
        ny = max(20, int((y2 - y1) / self.fds_config.grid_resolution))
        nz = max(20, int((z2 - z1) / self.fds_config.grid_resolution))
        fds_input = f"""! FDS Input File Generated by FlameProcessor
&HEAD CHID='fire_simulation' TITLE='Point Cloud Fire Simulation' /
&MESH IJK={nx},{ny},{nz} XB={x1:.3f},{x2:.3f},{y1:.3f},{y2:.3f},{z1:.3f},{z2:.3f} /
&TIME T_END={self.fds_config.simulation_time} /
&MISC TMPA={self.fds_config.ambient_temperature} /
&DUMP DT_DEVC={self.fds_config.output_interval} DT_HRR={self.fds_config.output_interval} /
&SURF ID='OPEN' RGB=255,255,255 /
&VENT XB={x1:.3f},{x1:.3f},{y1:.3f},{y2:.3f},{z1:.3f},{z2:.3f} SURF_ID='OPEN' /
&VENT XB={x2:.3f},{x2:.3f},{y1:.3f},{y2:.3f},{z1:.3f},{z2:.3f} SURF_ID='OPEN' /
&VENT XB={x1:.3f},{x2:.3f},{y1:.3f},{y1:.3f},{z1:.3f},{z2:.3f} SURF_ID='OPEN' /
&VENT XB={x1:.3f},{x2:.3f},{y2:.3f},{y2:.3f},{z1:.3f},{z2:.3f} SURF_ID='OPEN' /
&VENT XB={x1:.3f},{x2:.3f},{y1:.3f},{y2:.3f},{z2:.3f},{z2:.3f} SURF_ID='OPEN' /
&OBST XB={x1:.3f},{x2:.3f},{y1:.3f},{y2:.3f},{z1:.3f},{z1+0.01:.3f} SURF_ID='INERT' /
"""
        geometry = self.pointcloud_to_fds_geometry(pcd, material_labels)
        fds_input += geometry
        if ignition_point is None:
            ignition_point = ((x1 + x2) / 2, (y1 + y2) / 2, z1 + 0.1)
        ix, iy, iz = ignition_point
        fds_input += f"""
&SURF ID='IGNITOR' HRRPUA=500 RAMP_Q='IGNITION_RAMP' COLOR='RED' /
&RAMP ID='IGNITION_RAMP' T=0 F=0 /
&RAMP ID='IGNITION_RAMP' T=5 F=1 /
&RAMP ID='IGNITION_RAMP' T=10 F=0 /
&OBST XB={ix-0.1:.3f},{ix+0.1:.3f},{iy-0.1:.3f},{iy+0.1:.3f},{iz:.3f},{iz+0.1:.3f} SURF_ID='IGNITOR' /
&DEVC XYZ={ix:.3f},{iy:.3f},{iz+0.5:.3f} QUANTITY='TEMPERATURE' ID='TEMP_CENTER' /
&DEVC XYZ={ix:.3f},{iy:.3f},{iz+1.0:.3f} QUANTITY='VELOCITY' ID='VEL_CENTER' /
&SLCF PBZ={iz+0.5:.3f} QUANTITY='TEMPERATURE' /
&SLCF PBZ={iz+0.5:.3f} QUANTITY='HRRPUV' /
&SLCF PBY={(y1+y2)/2:.3f} QUANTITY='TEMPERATURE' /
&TAIL /
"""
        return fds_input

    def parse_fds_output(self, fds_results: Dict) -> Dict:
        """Parse FDS output to extract decay thresholds for materials."""
        decay_data = {}
        if 'hrr_data' in fds_results and fds_results['hrr_data']:
            try:
                # Assume hrr_data is a list of strings, each line: time,HRR
                for line in fds_results['hrr_data'][1:]:  # Skip header
                    time, hrr = map(float, line.split(','))
                    for material in self.fds_config.material_properties:
                        if material != 'A1':
                            if hrr <= 10.0:  # Threshold for burnout (kW/m²)
                                decay_data[material] = {'decay_time': time}
                                break
            except Exception as e:
                print(f"WARNING: Failed to parse FDS HRR data: {e}")
        
        if not decay_data:
            # Fallback to default burn durations
            for material, props in self.fds_config.material_properties.items():
                decay_data[material] = {'decay_time': props['burn_duration']}
        
        return decay_data

    def run_fds_simulation(self, fds_input: str) -> Dict:
        if not self.enable_fds or not self.fds_executable:
            return {'error': 'FDS not available'}
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, 'fire_simulation.fds')
            try:
                with open(input_file, 'w') as f:
                    f.write(fds_input)
            except Exception as e:
                return {'status': 'error', 'error': f'Failed to write input file: {str(e)}'}

            try:
                result = subprocess.run(
                    [self.fds_executable, input_file], 
                    capture_output=True, 
                    text=True, 
                    timeout=300
                )
                sim_results = {
                    'status': 'success' if result.returncode == 0 else 'failed',
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'output_files': [],
                    'hrr_data': [],
                    'temperature_data': []
                }
                if result.returncode != 0:
                    print(f"ERROR: FDS simulation failed: {result.returncode}")
                    return sim_results
                hrr_file = os.path.join(temp_dir, 'fire_simulation_hrr.csv')
                if os.path.exists(hrr_file):
                    try:
                        with open(hrr_file, 'r') as f:
                            lines = f.readlines()
                            data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
                            sim_results['hrr_data'] = data_lines
                    except Exception as e:
                        print(f"WARNING: Could not read HRR file: {e}")
                        sim_results['hrr_data'] = []

                devc_file = os.path.join(temp_dir, 'fire_simulation_devc.csv')
                if os.path.exists(devc_file):
                    with open(devc_file, 'r') as f:
                        try:
                            lines = f.readlines()
                            data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
                            sim_results['temperature_data'] = data_lines
                        except Exception as e:
                            print(f"WARNING: Could not read device file: {e}")
                            sim_results['temperature_data'] = []

                for file in os.listdir(temp_dir):
                    if file.startswith('fire_simulation') and file.endswith(('.csv', '.smv', '.s3d')):
                        sim_results['output_files'].append(file)
                return sim_results
            except subprocess.TimeoutExpired:
                return {'status': 'timeout', 'error': 'Simulation timed out after 5 minutes'}
            except Exception as e:
                return {'status': 'error', 'error': f'FDS execution failed: {str(e)}'}

    def create_fire_particles(self, pcd: o3d.geometry.PointCloud, material_labels: np.ndarray,
                            ignition_point: Tuple[float, float, float] = None) -> np.ndarray:
        points = np.array(pcd.points)
        if len(points) == 0:
            return np.array([]).reshape(0, 3)
        combustible_points = []
        for i, label in enumerate(material_labels):
            if 0 <= label < len(self.dms46):
                material_id = self.dms46[label]
                if material_id in self.flammability_map:
                    flam_class = self.flammability_map[material_id]
                    if flam_class != 'A1':
                        combustible_points.append(points[i])
        if not combustible_points:
            return np.array([]).reshape(0, 3)
        combustible_points = np.array(combustible_points)
        fire_particles = []
        for point in combustible_points:
            num_particles = np.random.randint(1, 4)
            for _ in range(num_particles):
                offset = np.random.normal(0, 0.05, 3)
                particle_pos = point + offset
                particle_pos[2] += np.random.uniform(0.1, 0.5)
                fire_particles.append(particle_pos)
        return np.array(fire_particles)

    def reset_simulation(self):
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread = None
        self.base_pointcloud = None
        self.base_material_labels = None
        self.fire_simulation_active = False
        self.current_simulation_time = 0.0
        self.ignition_point = None
        self.fire_history = []
        self.fire_states = {}
        self.fds_decay_data = {}
        self.base_frame_processed = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("DEBUG: Fire simulation reset")


    # In FlameProcessor class

    def _compute_fire_state(self, current_time: float) -> Dict:
        """Compute the fire propagation at a given time, with a forced ignition rule."""
        if self.base_pointcloud is None or self.base_material_labels is None or self.ignition_point is None:
            return {'error': 'Base simulation state not initialized.'}
        
        points = np.array(self.base_pointcloud.points)
        num_points = len(points)
        
        fire_intensity = np.zeros(num_points)
        fire_temperature = np.full(num_points, self.fds_config.ambient_temperature)
        burning_state = np.zeros(num_points, dtype=bool)
        burned_out = np.zeros(num_points, dtype=bool)

        # 1. FORCED IGNITION LOGIC (KICK-START)
        if self.forced_ignition_index is not None and current_time < self.time_step * 2:
            idx = self.forced_ignition_index
            burning_state[idx] = True
            fire_intensity[idx] = 0.5
            fire_temperature[idx] = 600.0

        # 2. PHYSICS-BASED PROPAGATION LOGIC
        propagation_speeds = {'A1': 0.0, 'A2': 0.02, 'B/C': 0.05, 'D/E': 0.15, 'F': 0.3}
        ignition_times = {'A1': float('inf'), 'A2': 30.0, 'B/C': 15.0, 'D/E': 5.0, 'F': 2.0}
        
        ignition_pos = np.array(self.ignition_point)
        distances_to_ignition = np.linalg.norm(points - ignition_pos, axis=1)

        for i in range(num_points):
            if burning_state[i]: continue

            label_idx = self.base_material_labels[i]
            if not (0 <= label_idx < len(self.dms46)): continue

            material_id = self.dms46[label_idx]
            flam_class = self.flammability_map.get(material_id)
            if not flam_class or flam_class == 'A1': continue
                
            prop_speed = propagation_speeds[flam_class]
            time_to_reach = distances_to_ignition[i] / max(prop_speed, 1e-6)
            actual_ignition_time = ignition_times[flam_class] + time_to_reach
            burn_duration_max = self.fds_decay_data.get(flam_class, {'decay_time': self.fds_config.material_properties[flam_class]['burn_duration']})['decay_time']
            
            if current_time >= actual_ignition_time and current_time <= actual_ignition_time + burn_duration_max:
                burning_state[i] = True
                burn_duration = current_time - actual_ignition_time
                max_intensity = {'A2': 0.3, 'B/C': 0.6, 'D/E': 0.9, 'F': 1.0}[flam_class]
                intensity = max_intensity * (1 - np.exp(-burn_duration / 10.0))
                fire_intensity[i] = min(intensity, max_intensity)
                max_temp = {'A2': 400, 'B/C': 600, 'D/E': 800, 'F': 1000}[flam_class]
                fire_temperature[i] = self.fds_config.ambient_temperature + (max_temp - self.fds_config.ambient_temperature) * intensity
            elif current_time > actual_ignition_time + burn_duration_max:
                burned_out[i] = True
        
        # 3. PARTICLE GENERATION
        burning_indices = np.where(burning_state)[0]
        fire_particles = []
        if len(burning_indices) > 0:
            burning_points = points[burning_indices]
            intensities = fire_intensity[burning_indices]
            for point, intensity in zip(burning_points, intensities):
                if intensity > 0.1:
                    num_particles = max(3, int(intensity * 30))
                    for _ in range(min(num_particles, 50)):
                        particle_offset = np.random.normal(0, 0.15 * intensity, 3)
                        particle_pos = point + particle_offset
                        particle_pos[2] += np.random.uniform(0.2, 1.0 * intensity)
                        fire_particles.append(particle_pos.tolist())

        smoke_particles = []
        if len(fire_particles) > 0:
            for fire_pos in fire_particles[:50]:
                smoke_pos = list(fire_pos).copy()
                smoke_pos[2] += np.random.uniform(0.5, 2.0)
                smoke_pos[0] += np.random.normal(0, 0.2)
                smoke_pos[1] += np.random.normal(0, 0.2)
                smoke_particles.append(smoke_pos)

        return {
            'fire_particles': fire_particles,
            'smoke_particles': smoke_particles,
            'burning_points': burning_indices.tolist(),
            'fire_intensities': fire_intensity.tolist(),
            'temperatures': fire_temperature.tolist(),
            'burned_out': burned_out.tolist(),
            'simulation_time': current_time,
            'total_burning_points': len(burning_indices),
            'max_temperature': float(np.max(fire_temperature)) if np.any(burning_state) else self.fds_config.ambient_temperature,
            'average_intensity': float(np.mean(fire_intensity[fire_intensity > 0])) if np.any(fire_intensity > 0) else 0.0
        }

    def _compute_fire_state1(self, current_time: float) -> Dict:
        """Compute the fire propagation at a given time."""
        if self.base_pointcloud is None or self.base_material_labels is None:
            return {'error': 'No base point cloud available'}
        points = np.array(self.base_pointcloud.points)
        propagation_speeds = {
            'A1': 0.0,
            'A2': 0.02,
            'B/C': 0.05,
            'D/E': 0.15,
            'F': 0.3
        }
        ignition_times = {
            'A1': float('inf'),
            'A2': 30.0,
            'B/C': 15.0,
            'D/E': 5.0,
            'F': 2.0
        }
        fire_intensity = np.zeros(len(points))
        fire_temperature = np.full(len(points), 20.0)
        burning_state = np.zeros(len(points), dtype=bool)
        burned_out = np.zeros(len(points), dtype=bool)
        if self.ignition_point is None:
            min_bound = self.base_pointcloud.get_min_bound()
            max_bound = self.base_pointcloud.get_max_bound()
            self.ignition_point = ((min_bound[0] + max_bound[0]) / 2,
                                 (min_bound[1] + max_bound[1]) / 2,
                                 min_bound[2] + 0.1)
        ignition_pos = np.array(self.ignition_point)
        distances_to_ignition = np.linalg.norm(points - ignition_pos, axis=1)
        for i, (point, label) in enumerate(zip(points, self.base_material_labels)):
            if 0 <= label < len(self.dms46):
                material_id = self.dms46[label]
                if material_id in self.flammability_map:
                    flam_class = self.flammability_map[material_id]
                    distance_to_ignition = distances_to_ignition[i]
                    ignition_time = ignition_times[flam_class]
                    prop_speed = propagation_speeds[flam_class]
                    time_to_reach = distance_to_ignition / max(prop_speed, 0.001) if prop_speed > 0 else float('inf')
                    actual_ignition_time = ignition_time + time_to_reach
                    burn_duration_max = self.fds_decay_data.get(flam_class, {'decay_time': self.fds_config.material_properties[flam_class]['burn_duration']})['decay_time']
                    if current_time >= actual_ignition_time and current_time <= actual_ignition_time + burn_duration_max and flam_class != 'A1':
                        burning_state[i] = True
                        burn_duration = current_time - actual_ignition_time
                        max_intensity = {'A2': 0.3, 'B/C': 0.6, 'D/E': 0.9, 'F': 1.0}[flam_class]
                        intensity = max_intensity * (1 - np.exp(-burn_duration / 10.0))
                        fire_intensity[i] = min(intensity, max_intensity)
                        max_temp = {'A2': 400, 'B/C': 600, 'D/E': 800, 'F': 1000}[flam_class]
                        fire_temperature[i] = 20 + (max_temp - 20) * intensity
                    elif current_time > actual_ignition_time + burn_duration_max and flam_class != 'A1':
                        burned_out[i] = True
        burning_indices = np.where(burning_state)[0]
        fire_particles = []
        if len(burning_indices) > 0:
            burning_points = points[burning_indices]
            intensities = fire_intensity[burning_indices]
            for point, intensity in zip(burning_points, intensities):
                if intensity > 0.1:
                    num_particles = max(1, int(intensity * 30))
                    for _ in range(min(num_particles, 50)):
                        particle_offset = np.random.normal(0, 0.15 * intensity, 3)
                        particle_pos = point + particle_offset
                        particle_pos[2] += np.random.uniform(0.2, 0.5 * intensity)
                        fire_particles.append(particle_pos.tolist())
        smoke_particles = []
        if len(fire_particles) > 0:
            for fire_pos in fire_particles[:50]:
                smoke_pos = list(fire_pos).copy()
                smoke_pos[2] += np.random.uniform(0.5, 2.0)
                smoke_pos[0] += np.random.normal(0, 0.2)
                smoke_pos[1] += np.random.normal(0, 0.2)
                smoke_particles.append(smoke_pos)
        return {
            'fire_particles': fire_particles,
            'smoke_particles': smoke_particles,
            'burning_points': burning_indices.tolist(),
            'fire_intensities': fire_intensity.tolist(),
            'temperatures': fire_temperature.tolist(),
            'burned_out': burned_out.tolist(),
            'simulation_time': current_time,
            'total_burning_points': len(burning_indices),
            'max_temperature': float(np.max(fire_temperature)),
            'average_intensity': float(np.mean(fire_intensity[fire_intensity > 0])) if np.any(fire_intensity > 0) else 0.0
        }

    def _precompute_fire_states(self):
        current_time = 0.0
        while current_time <= self.max_simulation_time and self.fire_simulation_active:
            time.sleep(0.1)
            with self.state_lock:
                if current_time not in self.fire_states:
                    state = self._compute_fire_state(current_time)
                    self.fire_states[current_time] = state
                    print(f"DEBUG: Precomputed fire state at time {current_time:.1f}s")
            current_time += self.time_step
        print("DEBUG: Precomputation thread finished")

    def simulate_fire_propagation(self, current_time: float) -> Dict:
        if self.base_pointcloud is None or self.base_material_labels is None:
            return {'error': 'No base point cloud available'}
        if self.fire_simulation_active and (self.simulation_thread is None or not self.simulation_thread.is_alive()):
            self.simulation_thread = threading.Thread(target=self._precompute_fire_states, daemon=True)
            self.simulation_thread.start()
            print("DEBUG: Started precomputation")
        with self.state_lock:
            available_times = sorted(self.fire_states.keys())
            if not available_times:
                print(f"DEBUG: No precomputed state for time {current_time:.1f}s")
                return self._compute_fire_state(current_time)
            closest_time = min(available_times, key=lambda t: abs(t - current_time))
            if abs(closest_time - current_time) <= self.time_step / 2:
                print(f"DEBUG: Using precomputed state at {closest_time:.1f}s")
                return self.fire_states[closest_time]
            else:
                timeout = 5.0
                start_time = time.time()
                while current_time not in self.fire_states and time.time() - start_time < timeout:
                    time.sleep(0.1)
                with self.state_lock:
                    if current_time in self.fire_states:
                        print(f"DEBUG: Retrieved precomputed state at {current_time:.1f}s")
                        return self.fire_states[current_time]
                    else:
                        print(f"DEBUG: Timeout, computing synchronously for {current_time:.1f}s")
                        return self._compute_fire_state(current_time)

    def create_point_cloud_with_fire(self, fire_sim_result: Dict) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        fire_particles = np.array(fire_sim_result.get('fire_particles', []))
        smoke_particles = np.array(fire_sim_result.get('smoke_particles', []))
        all_points = []
        all_colors = []
        if len(fire_particles) > 0:
            all_points.extend(fire_particles)
            fire_colors = np.array([[1.0, 0.5, 0.0] for _ in range(len(fire_particles))])
            all_colors.extend(fire_colors)
        if len(smoke_particles) > 0:
            all_points.extend(smoke_particles)
            smoke_colors = np.full((len(smoke_particles), 3), [0.5, 0.5, 0.5])
            all_colors.extend(smoke_colors)
        if all_points:
            pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        return pcd

    def process_frame(self, frame: np.ndarray, ignition_point: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Processes a video frame to generate and/or update a fire simulation.
        Conforms to the BaseProcessor contract by returning a static image and a dynamic data payload.

        Returns:
            Tuple[np.ndarray, Dict]: 
            - The first element is the initial video frame, serving as a static background image.
            - The second element is the data payload, containing the updated, serializable point cloud.
        """
        if not self.base_frame_processed:
            print("DEBUG: Processing base frame...")

            # _process_base_frame sets internal state (self.base_pointcloud, etc.)
            base_pcd, result_payload = self._process_base_frame(frame, ignition_point)
            self.base_frame_processed = True
            self.fire_simulation_active = True

            if self.simulation_thread is None or not self.simulation_thread.is_alive():
                self.simulation_thread = threading.Thread(target=self._precompute_fire_states, daemon=True)
                self.simulation_thread.start()
            
            # The payload for the first frame should contain the initial point cloud.
            result_payload['point_cloud'] = self.pointcloud_to_json(base_pcd, self.base_material_labels)
            
            # Return the static frame and the initial data payload.
            return self.vis_bgr, result_payload

        # --- Logic for subsequent frames ---
        print(f"DEBUG: Fire propagation at time {self.current_simulation_time:.1f}s")
        self.current_simulation_time += self.time_step

        fire_sim_result = self.simulate_fire_propagation(self.current_simulation_time)
        fire_effects_pcd = self.create_point_cloud_with_fire(fire_sim_result)
        
        scene_pcd_with_fire_effects = o3d.geometry.PointCloud(self.base_pointcloud)
        updated_scene_data = self.pointcloud_to_json(
            scene_pcd_with_fire_effects, 
            intensities=fire_sim_result.get('fire_intensities'),
            burned_out=fire_sim_result.get('burned_out')
        )
        scene_pcd_with_fire_effects.colors = o3d.utility.Vector3dVector(updated_scene_data['colors'])

        combined_pcd_for_vis = scene_pcd_with_fire_effects + fire_effects_pcd
        
        # The data payload is a dictionary containing the new, combined point cloud and metadata.
        result_payload = {
            'simulation_time': self.current_simulation_time,
            'fire_active': fire_sim_result.get('total_burning_points', 0) > 0,
            'simulation_summary': {
                'total_burning_points': fire_sim_result.get('total_burning_points', 0),
                'max_temperature': fire_sim_result.get('max_temperature', self.fds_config.ambient_temperature),
                'average_intensity': fire_sim_result.get('average_intensity', 0.0)
            },
        }
        
        print(f"DEBUG: Fire details : {result_payload}")

        # History tracking
        self.fire_history.append({
            'time': self.current_simulation_time,
            'summary': result_payload['simulation_summary'] 
        })
        if len(self.fire_history) > 100:
            self.fire_history = self.fire_history[-100:]
        result_payload['fire_history'] = self.fire_history[-10:]

        # Return the original static frame and the updated data payload.
        return self.vis_bgr, self.pointcloud_to_json(combined_pcd_for_vis)

    def process_frame1(self, frame: np.ndarray, ignition_point: Tuple[float, float, float] = None) -> Tuple[o3d.geometry.PointCloud, Dict]:
        if not self.base_frame_processed:
            print("DEBUG: Processing base frame...")
            # _process_base_frame already sets self.base_pointcloud and returns the initial data
            base_pcd, pointcloud_data = self._process_base_frame(frame, ignition_point)
            self.base_frame_processed = True
            self.fire_simulation_active = True

            if self.simulation_thread is None or not self.simulation_thread.is_alive():
                self.simulation_thread = threading.Thread(target=self._precompute_fire_states, daemon=True)
                self.simulation_thread.start()
            
            # The initial point cloud to show is the base scene itself
            # The JSON payload contains all the details including the initial fire state (which is likely none)
            return base_pcd, pointcloud_data

        print(f"DEBUG: Fire propagation at time {self.current_simulation_time:.1f}s")
        self.current_simulation_time += self.time_step

        # Get the latest fire simulation state (particles, intensities, etc.)
        fire_sim_result = self.simulate_fire_propagation(self.current_simulation_time)

        # Create a point cloud of just the fire and smoke particles
        fire_effects_pcd = self.create_point_cloud_with_fire(fire_sim_result)
        
        # The base point cloud's colors need to be updated to show burning/charring
        # We can create a temporary copy to modify its colors without altering the original
        scene_pcd_with_fire_effects = o3d.geometry.PointCloud(self.base_pointcloud)
        
        # Get the modified colors for the scene based on the fire simulation
        # The pointcloud_to_json function can be leveraged to calculate this
        updated_scene_data = self.pointcloud_to_json(
            scene_pcd_with_fire_effects, 
            intensities=fire_sim_result.get('fire_intensities'),
            burned_out=fire_sim_result.get('burned_out')
        )
        scene_pcd_with_fire_effects.colors = o3d.utility.Vector3dVector(updated_scene_data['colors'])

        # Create a final, combined point cloud for easy visualization by the client
        combined_pcd_for_vis = scene_pcd_with_fire_effects + fire_effects_pcd
        
        # Structure the detailed result payload for the client
        result_dict = {
            'simulation_time': self.current_simulation_time,
            'fire_active': fire_sim_result.get('total_burning_points', 0) > 0,
            'frame_type': 'fire_propagation',
            
            # This contains the original scene geometry, but with colors updated to show fire damage
            'scene_pointcloud': self.pointcloud_to_json(
                self.base_pointcloud, self.base_material_labels, 
                fire_sim_result.get('fire_intensities'), 
                fire_sim_result.get('temperatures'), 
                fire_sim_result.get('burned_out')
            ),

            # This contains only the fire/smoke particles
            'fire_effects_pointcloud': self.pointcloud_to_json(fire_effects_pcd),

            # High-level summary of the simulation state
            'simulation_summary': {
                'total_burning_points': fire_sim_result.get('total_burning_points', 0),
                'max_temperature': fire_sim_result.get('max_temperature', self.fds_config.ambient_temperature),
                'average_intensity': fire_sim_result.get('average_intensity', 0.0)
            },
        }

        self.fire_history.append({
            'time': self.current_simulation_time,
            'summary': result_dict['simulation_summary'] 
        })
        if len(self.fire_history) > 100:
            self.fire_history = self.fire_history[-100:]
        result_dict['fire_history'] = self.fire_history[-10:]

        return combined_pcd_for_vis, result_dict
    
    # In FlameProcessor class

    def _find_worst_case_ignition_point(self, pcd: o3d.geometry.PointCloud) -> Tuple[float, float, float]:
        """
        Finds the best ignition point, guarantees a result, and stores the point's index.
        It prioritizes the most vulnerable material, falling back to the geometric center if needed.
        """
        points = np.asarray(pcd.points)
        vulnerability_order = ['F', 'D/E', 'B/C', 'A2']

        # Find the best point based on material
        for flam_class in vulnerability_order:
            target_indices = [i for i, label_idx in enumerate(self.base_material_labels) if 0 <= label_idx < len(self.dms46) and self.flammability_map.get(self.dms46[label_idx]) == flam_class]
            
            if target_indices:
                target_points = points[target_indices]
                centroid = np.mean(target_points, axis=0)
                
                pcd_subset = o3d.geometry.PointCloud()
                pcd_subset.points = o3d.utility.Vector3dVector(target_points)
                kdtree = o3d.geometry.KDTreeFlann(pcd_subset)
                [_, idx, _] = kdtree.search_knn_vector_3d(centroid, 1)

                self.forced_ignition_index = target_indices[idx[0]]
                chosen_point_coords = points[self.forced_ignition_index]
                
                print(f"DEBUG: Found worst-case ignition point in material '{flam_class}' at index {self.forced_ignition_index}")
                return tuple(chosen_point_coords)

        # Fallback: No combustible materials found
        print("DEBUG: No combustible materials found. Forcing ignition at geometric center.")
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.get_center(), 1)
        
        self.forced_ignition_index = idx[0]
        chosen_point_coords = points[self.forced_ignition_index]
        print(f"DEBUG: Using fallback ignition point at index {self.forced_ignition_index}")
        
        return tuple(chosen_point_coords)
    
    def _find_worst_case_ignition_point1(self, pcd: o3d.geometry.PointCloud, material_labels: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Finds an ignition point by identifying the most vulnerable material present.
        It calculates the centroid of all points belonging to the highest-risk class.
        """
        # Define the order of vulnerability, from most to least flammable.
        vulnerability_order = ['F', 'D/E', 'B/C', 'A2']
        points = np.asarray(pcd.points)

        for flam_class in vulnerability_order:
            # Find the indices of all points that belong to the current flammability class.
            target_indices = []
            for i, label_idx in enumerate(material_labels):
                if 0 <= label_idx < len(self.dms46):
                    material_id = self.dms46[label_idx]
                    if self.flammability_map.get(material_id) == flam_class:
                        target_indices.append(i)

            # If we found any points for this class, this is our target.
            if target_indices:
                # Get all points belonging to this high-risk class.
                target_points = points[target_indices]
                # Calculate the centroid (average position) of these points.
                centroid = np.mean(target_points, axis=0)
                print(f"DEBUG: Found worst-case ignition point in material '{flam_class}' at {centroid.tolist()}")
                return tuple(centroid)

        # If no combustible materials were found at all.
        print("DEBUG: No combustible materials found to select an automatic ignition point.")
        return None

    def _process_base_frame(self, frame: np.ndarray, ignition_point: Optional[Tuple[float, float, float]] = None) -> Tuple[Optional[o3d.geometry.PointCloud], Dict]:
        print(f"DEBUG: _process_base_frame shape: {frame.shape}")
        h, w = frame.shape[:2]; scale = 512 / max(h, w)
        new_h, new_w = math.ceil(scale * h), math.ceil(scale * w)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float()
        value_scale = 255; mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
        normalize = TTR.Normalize([m * value_scale for m in mean], [s * value_scale for s in std])
        image = normalize(image).unsqueeze(0).to(self.device)
        try:
            with torch.no_grad():
                prediction = self.material_model(image)[0].detach().cpu().numpy()[0, 0].astype(np.int64)
            pil_image, _ = self.preprocess_image(resized_frame)
            image_tensor = self.depth_transform(pil_image).to(self.device)
            if image_tensor.dtype == torch.float16 and self.device.type == 'cpu': image_tensor = image_tensor.float()
            with torch.no_grad():
                prediction_depth = self.depth_model.infer(image_tensor.unsqueeze(0))
            depth_np = prediction_depth["depth"].squeeze().detach().cpu().numpy()
            focallength_px = prediction_depth["focallength_px"].squeeze()
            
            pcd, material_labels = self.create_point_cloud_with_materials(depth_np, resized_frame, prediction, focallength_px)
            
            self.base_pointcloud = pcd
            self.base_material_labels = material_labels

            if ignition_point is not None:
                kdtree = o3d.geometry.KDTreeFlann(self.base_pointcloud)
                [_, idx, _] = kdtree.search_knn_vector_3d(ignition_point, 1)
                self.forced_ignition_index = idx[0]
                self.ignition_point = tuple(self.base_pointcloud.points[self.forced_ignition_index])
                print(f"DEBUG: Using user-specified ignition point, nearest point is at index {self.forced_ignition_index}")
            else:
                self.ignition_point = self._find_worst_case_ignition_point(self.base_pointcloud)

            initial_fire = self._compute_fire_state(0.0)
            pointcloud_json = self.pointcloud_to_json(pcd, material_labels)
            pointcloud_json['initial_fire_simulation'] = initial_fire
            
            return pcd, pointcloud_json

        except Exception as e:
            import traceback
            print(f"ERROR: Depth processing failed: {e}\n{traceback.format_exc()}")
            self.base_pointcloud = None
            self.base_material_labels = None
            return None, {'error': str(e)}

    def _process_base_frame1(self, frame: np.ndarray, ignition_point: Optional[Tuple[float, float, float]] = None) -> Tuple[o3d.geometry.PointCloud, Dict]:
        print(f"DEBUG: _process_base_frame shape: {frame.shape}")
        h, w = frame.shape[:2]
        scale = 512 / max(h, w)
        new_h = math.ceil(scale * h)
        new_w = math.ceil(scale * w)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float()
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        normalize = TTR.Normalize(mean, std)
        image = normalize(image)
        image = image.unsqueeze(0).to(self.device)
        try:
            with torch.no_grad():
                model_output = self.material_model(image)
                prediction = model_output[0].detach().cpu().numpy()[0, 0].astype(np.int64)
        except Exception as e:
            print(f"ERROR: Material inference failed: {e}")
            prediction = np.zeros((new_h, new_w), dtype=np.int64)
        self.vis_bgr = self.apply_color(prediction)
        unique_labels = np.unique(prediction)
        detected_classes = []
        for label in unique_labels:
            if 0 <= label < len(self.dms46):
                material_id = self.dms46[label]
                if material_id in self.flammability_map:
                    detected_classes.append(self.flammability_map[material_id])
        detected_classes = sorted(set(detected_classes))
        result_text = "Detected classes: " + ", ".join(detected_classes) if detected_classes else "No materials detected"
        pil_image, _ = self.preprocess_image(resized_frame)
        try:
            image_tensor = self.depth_transform(pil_image).to(self.device)
            if image_tensor.dtype == torch.float16 and self.device == 'cpu':
                image_tensor = image_tensor.float()
            with torch.no_grad():
                prediction_depth = self.depth_model.infer(image_tensor.unsqueeze(0))
            depth_np = prediction_depth["depth"].squeeze().detach().cpu().numpy()
            focallength_px = prediction_depth["focallength_px"].squeeze()
            pcd, material_labels = self.create_point_cloud_with_materials(depth_np, resized_frame, prediction, focallength_px)
            self.base_pointcloud = pcd
            self.base_material_labels = material_labels
            
            # --- START OF MODIFICATION ---
            # Set the ignition point. Prioritize a user-provided point.
            if ignition_point is not None:
                self.ignition_point = ignition_point
                print(f"DEBUG: Using user-specified ignition point: {self.ignition_point}")
            else:
                # If no point is provided, automatically find the worst-case spot.
                self.ignition_point = self._find_worst_case_ignition_point(pcd, material_labels)

            # If no point was found (e.g., no combustible materials), fall back to the center.
            if self.ignition_point is None:
                print("DEBUG: Fallback to central ignition point.")
                min_b, max_b = pcd.get_min_bound(), pcd.get_max_bound()
                self.ignition_point = ((min_b[0] + max_b[0]) / 2, (min_b[1] + max_b[1]) / 2, min_b[2] + 0.1)
            # --- END OF MODIFICATION ---

            pointcloud_json = self.pointcloud_to_json(pcd, material_labels)
            if self.enable_fds and len(pcd.points) > 100:
                print("DEBUG: Running FDS simulation...")
                try:
                    # Pass the determined ignition point to the FDS simulation
                    fds_input = self.create_fds_input_file(pcd, material_labels, self.ignition_point)
                    fds_results = self.run_fds_simulation(fds_input)
                    self.fds_decay_data = self.parse_fds_output(fds_results)
                    pointcloud_json['fds_simulation'] = fds_results
                    self.fire_simulation_active = True
                    print("DEBUG: FDS simulation completed")
                except Exception as e:
                    print(f"ERROR: FDS simulation failed: {e}")
                    pointcloud_json['fds_simulation'] = {'status': 'error', 'error': str(e)}
            
            # The initial fire state will now be calculated relative to the new ignition point
            initial_fire = self._compute_fire_state(0.0)
            pointcloud_json['initial_fire_simulation'] = initial_fire
        except Exception as e:
            print(f"ERROR: Depth processing failed: {e}")
            pointcloud_json = {
                'error': str(e), 
                'points': [], 
                'colors': [], 
                'normals': [],
                'material_labels': [],
                'initial_fire_simulation': {}
            }
            pcd = o3d.geometry.PointCloud()
        pointcloud_json['detected_flammability_classes'] = detected_classes
        pointcloud_json['result_text'] = result_text
        pointcloud_json['frame_type'] = 'base_pointcloud'
        return pcd, pointcloud_json

    def process_pointcloud(self, point_cloud_data: Dict, ignition_point: Tuple[float, float, float] = None) -> Tuple[Optional[o3d.geometry.PointCloud], Union[str, Dict]]:
        if not point_cloud_data or 'points' not in point_cloud_data:
            return None, {"error": "Invalid point cloud"}
        try:
            points = np.array(point_cloud_data['points'])
            colors = np.array(point_cloud_data.get('colors', []))
            material_labels = np.array(point_cloud_data.get('material_labels', []))
            if len(points) == 0:
                return None, {"error": "Empty point cloud"}
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if len(colors) > 0:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            if len(material_labels) == 0:
                material_labels = np.full(len(points), 52)
            self.base_pointcloud = pcd
            self.base_material_labels = material_labels
            self.ignition_point = ignition_point
            self.base_frame_processed = True
            self.fire_simulation_active = True
            self.current_simulation_time = 0.0
            if self.enable_fds and len(pcd.points) > 100:
                fds_input = self.create_fds_input_file(pcd, material_labels, ignition_point)
                fds_results = self.run_fds_simulation(fds_input)
                self.fds_decay_data = self.parse_fds_output(fds_results)
            initial_fire = self._compute_fire_state(0.0)
            enhanced_data = point_cloud_data.copy()
            enhanced_data['fire_simulation'] = initial_fire
            enhanced_data['frame_type'] = 'base_pointcloud_from_data'
            enhanced_data['simulation_active'] = True
            visualization_pcd = self.create_point_cloud_with_fire(initial_fire)
            if self.simulation_thread is None or not self.simulation_thread.is_alive():
                self.simulation_thread = threading.Thread(target=self._precompute_fire_states, daemon=True)
                self.simulation_thread.start()
            return visualization_pcd, {'status': 'success', 'data': enhanced_data}
        except Exception as e:
            print(f"ERROR: Point cloud processing failed: {e}")
            return None, {"error": str(e)}

    def get_next_fire_frame(self) -> Tuple[Optional[o3d.geometry.PointCloud], Dict]:
        if not self.fire_simulation_active or self.base_pointcloud is None:
            return None, {"error": "No active simulation"}
        self.current_simulation_time += self.time_step
        fire_sim_result = self.simulate_fire_propagation(self.current_simulation_time)
        visualization_pcd = self.create_point_cloud_with_fire(fire_sim_result)
        result_dict = {
            'simulation_time': self.current_simulation_time,
            'fire_simulation': fire_sim_result,
            'base_pointcloud': self.pointcloud_to_json(
                self.base_pointcloud, 
                self.base_material_labels,
                fire_sim_result.get('fire_intensities'),
                fire_sim_result.get('temperatures'), 
                fire_sim_result.get('burned_out')
            ),
            'frame_type': 'fire_propagation_only',
            'fire_active': fire_sim_result.get('total_burning_points', 0) > 0
        }
        self.fire_history.append({
            'time': self.current_simulation_time,
            'fire_data': fire_sim_result
        })
        if len(self.fire_history) > 100:
            self.fire_history = self.fire_history[-100:]
        result_dict['fire_history'] = self.fire_history[-5:]
        return visualization_pcd, result_dict

    def set_simulation_parameters(self, time_step: float = None, ignition_point: Tuple[float, float, float] = None):
        if time_step is not None:
            self.time_step = time_step
        if ignition_point is not None:
            self.ignition_point = ignition_point

    def get_simulation_status(self) -> Dict:
        with self.state_lock:
            return {
                'base_frame_processed': self.base_frame_processed,
                'fire_simulation_active': self.fire_simulation_active,
                'current_simulation_time': self.current_simulation_time,
                'time_step': self.time_step,
                'ignition_point': self.ignition_point,
                'base_pointcloud_size': len(self.base_pointcloud.points) if self.base_pointcloud else 0,
                'history_length': len(self.fire_history),
                'precomputed_times': sorted(self.fire_states.keys())
            }

    def get_material_statistics(self, material_labels: np.ndarray) -> Dict:
        stats = {
            'total_points': len(material_labels),
            'flammability_distribution': {},
            'fire_risk': {
                'risk_level': 'LOW',
                'combustible_percentage': 0.0,
                'high_risk_percentage': 0.0
            }
        }
        flammability_counts = {}
        for label in material_labels:
            if 0 <= label < len(self.dms46):
                material_id = self.dms46[label]
                if material_id in self.flammability_map:
                    flam_class = self.flammability_map[material_id]
                    flammability_counts[flam_class] = flammability_counts.get(flam_class, 0) + 1
        
        total_points = len(material_labels)
        if total_points > 0:
            for flam_class, count in flammability_counts.items():
                percentage = (count / total_points) * 100
                stats['flammability_distribution'][flam_class] = {
                    'count': count,
                    'percentage': percentage
                }
        
        combustible_classes = ['A2', 'B/C', 'D/E', 'F']
        high_risk_classes = ['D/E', 'F']
        combustible_count = sum(flammability_counts.get(cls, 0) for cls in combustible_classes)
        high_risk_count = sum(flammability_counts.get(cls, 0) for cls in high_risk_classes)
        if total_points > 0:
            combustible_percentage = (combustible_count / total_points) * 100
            high_risk_percentage = (high_risk_count / total_points) * 100
            stats['fire_risk']['combustible_percentage'] = combustible_percentage
            stats['fire_risk']['high_risk_percentage'] = high_risk_percentage
            if high_risk_percentage > 50:
                stats['fire_risk']['risk_level'] = 'EXTREME'
            elif high_risk_percentage > 25:
                stats['fire_risk']['risk_level'] = 'HIGH'
            elif combustible_percentage > 50:
                stats['fire_risk']['risk_level'] = 'MODERATE'
            elif combustible_percentage >= 10:
                stats['fire_risk']['risk_level'] = 'LOW'
            else:
                stats['fire_risk']['risk_level'] = 'MINIMAL'
        
        return stats

processor = FlameProcessor(
    enable_fds=True,
    time_step=0.5,
    fds_executable=os.getenv('FDS_EXECUTABLE', None)
)
app = processor.app
