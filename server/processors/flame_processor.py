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
from .base_processor import BaseProcessor
from depth_pro import create_model_and_transforms

class FDSConfig:
    """Configuration class for FDS simulation parameters."""
    
    def __init__(self):
        # Material properties for FDS (based on flammability classes)
        self.material_properties = {
            'A1': {  # Non-combustible (concrete, steel, glass)
                'density': 2300,  # kg/m³
                'conductivity': 1.6,  # W/m·K
                'specific_heat': 1000,  # J/kg·K
                'emissivity': 0.9,
                'ignition_temperature': None,  # Non-combustible
                'heat_of_combustion': 0,  # MJ/kg
                'soot_yield': 0,
                'co_yield': 0,
                'color': 'GRAY'
            },
            'A2': {  # Limited combustibility (treated wood, some composites)
                'density': 600,
                'conductivity': 0.15,
                'specific_heat': 1500,
                'emissivity': 0.85,
                'ignition_temperature': 300,  # °C
                'heat_of_combustion': 10,  # MJ/kg
                'soot_yield': 0.01,
                'co_yield': 0.004,
                'color': 'BROWN'
            },
            'B/C': {  # Combustible (wood, paper, some plastics)
                'density': 500,
                'conductivity': 0.12,
                'specific_heat': 1600,
                'emissivity': 0.8,
                'ignition_temperature': 250,  # °C
                'heat_of_combustion': 18,  # MJ/kg
                'soot_yield': 0.02,
                'co_yield': 0.008,
                'color': 'ORANGE'
            },
            'D/E': {  # Highly combustible (synthetic materials, foam)
                'density': 400,
                'conductivity': 0.08,
                'specific_heat': 1400,
                'emissivity': 0.85,
                'ignition_temperature': 200,  # °C
                'heat_of_combustion': 25,  # MJ/kg
                'soot_yield': 0.15,
                'co_yield': 0.05,
                'color': 'RED'
            },
            'F': {  # Extremely combustible (accelerants, highly flammable liquids)
                'density': 300,
                'conductivity': 0.05,
                'specific_heat': 1200,
                'emissivity': 0.9,
                'ignition_temperature': 150,  # °C
                'heat_of_combustion': 35,  # MJ/kg
                'soot_yield': 0.25,
                'co_yield': 0.08,
                'color': 'MAGENTA'
            }
        }
        
        # Default simulation parameters
        self.simulation_time = 60.0  # seconds
        self.output_interval = 1.0  # seconds
        self.grid_resolution = 0.1  # meters
        self.ambient_temperature = 20.0  # °C
        
    def get_hrr_per_unit_area(self, flammability_class: str) -> float:
        """Get heat release rate per unit area for material class."""
        hrr_map = {
            'A1': 0,      # Non-combustible
            'A2': 50,     # kW/m²
            'B/C': 200,   # kW/m²
            'D/E': 500,   # kW/m²
            'F': 1000     # kW/m²
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
                 enable_fds: bool = True):
        """
        Initialize the FlameProcessor with material segmentation, depth estimation, and FDS integration.

        Args:
            jit_path: Path to the pre-trained material segmentation model.
            taxonomy_path: Path to the taxonomy JSON file.
            depth_checkpoint_uri: Path to the depth estimation model checkpoint.
            use_gpu: Whether to use GPU if available.
            voxel_size: Voxel size for point cloud downsampling.
            estimate_normals: Whether to estimate normals for the point cloud.
            fds_executable: Path to FDS executable (if None, will search in PATH).
            enable_fds: Whether to enable FDS simulation.
        """
        super().__init__()

        # Initialize FDS configuration
        self.fds_config = FDSConfig()
        self.enable_fds = enable_fds
        self.fds_executable = fds_executable or self._find_fds_executable()
        
        # Material Segmentation Initialization (keeping your existing code)
        if jit_path is None:
            jit_path = '/home/znasif/vidServer/server/models/DMS46_v1.pt'
        if taxonomy_path is None:
            taxonomy_path = '/home/znasif/vidServer/server/models/taxonomy.json'

        print(f"DEBUG: Loading material model from: {jit_path}")
        print(f"DEBUG: Loading taxonomy from: {taxonomy_path}")

        # Load taxonomy
        try:
            with open(taxonomy_path, 'rb') as f:
                t = json.load(f)
            print(f"DEBUG: Taxonomy loaded successfully. Keys: {list(t.keys())}")
        except Exception as e:
            print(f"ERROR: Failed to load taxonomy: {e}")
            raise

        # Define DMS46 indices
        dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                 46, 47, 48, 49, 50, 51, 52, 53, 56]
        print(f"DEBUG: DMS46 indices: {len(dms46)} materials")

        # Material names
        self.material_names = [t['names'][i] for i in dms46]
        print(f"DEBUG: Material names: {self.material_names}")

        # Original colormap
        original_colormap = np.array([t['srgb_colormap'][i] for i in dms46], dtype=np.uint8)
        print(f"DEBUG: Original colormap shape: {original_colormap.shape}")

        # Flammability mapping
        self.flammability_map = {
            1: 'A1', 2: 'B/C', 3: 'A1', 4: 'D/E', 5: 'D/E', 6: 'B/C', 7: 'A1', 8: 'B/C',
            9: 'D/E', 10: 'A1', 11: 'B/C', 12: 'A1', 13: 'D/E', 15: 'F', 16: 'D/E', 17: 'D/E',
            18: 'D/E', 19: 'D/E', 20: 'A2', 21: 'D/E', 23: 'D/E', 24: 'D/E', 26: 'A1',
            27: 'A2', 29: 'B/C', 30: 'D/E', 32: 'D/E', 33: 'D/E', 34: 'D/E', 35: 'D/E',
            36: 'D/E', 37: 'D/E', 38: 'D/E', 39: 'D/E', 41: 'D/E', 43: 'A1', 44: 'A1',
            46: 'A1', 47: 'B/C', 48: 'D/E', 49: 'D/E', 50: 'B/C', 51: 'D/E', 52: 'D/E',
            53: 'D/E', 56: 'D/E'
        }
        print(f"DEBUG: Flammability mapping created for {len(self.flammability_map)} materials")

        # Flammability class colors
        try:
            self.class_colors = {
                'A1': original_colormap[dms46.index(26)],  # Yellow [225, 225, 0]
                'A2': original_colormap[dms46.index(20)],  # Cyan [0, 137, 188]
                'B/C': original_colormap[dms46.index(29)],  # Pink [225, 137, 188]
                'D/E': original_colormap[dms46.index(52)],  # Dark red [137, 0, 0]
                'F': original_colormap[dms46.index(15)]    # Light red [225, 188, 188]
            }
            print(f"DEBUG: Class colors: {self.class_colors}")
        except Exception as e:
            print(f"ERROR: Failed to create class colors: {e}")
            raise

        # SRGB colormap for flammability classes
        self.srgb_colormap = np.array([self.class_colors[self.flammability_map[i]] for i in dms46], dtype=np.uint8)
        print(f"DEBUG: SRGB colormap shape: {self.srgb_colormap.shape}")

        # Load material segmentation model
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"DEBUG: Using device: {self.device}")
        try:
            self.material_model = torch.jit.load(jit_path, map_location=self.device)
            self.material_model.eval()
            print("DEBUG: Material model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load material model: {e}")
            raise

        # Depth Estimation Initialization
        if depth_checkpoint_uri is None:
            depth_checkpoint_uri = '/home/znasif/vision-depth-pro/checkpoints/depth_pro.pt'
        self.depth_checkpoint_uri = depth_checkpoint_uri
        self.voxel_size = voxel_size
        self.estimate_normals = estimate_normals
        self._load_depth_model()

    def _find_fds_executable(self) -> Optional[str]:
        """Find FDS executable in system PATH."""
        import shutil
        fds_names = ['fds', 'fds.exe', 'fds6']
        for name in fds_names:
            path = shutil.which(name)
            if path:
                print(f"DEBUG: Found FDS executable at: {path}")
                return path
        print("WARNING: FDS executable not found in PATH")
        return None

    def _load_depth_model(self):
        """Load the depth estimation model."""
        self.depth_model, self.depth_transform = create_model_and_transforms(
            device=self.device,
            precision=torch.half if self.device == 'cuda' else torch.float32
        )
        self.depth_model.eval()
        print(f"DEBUG: Depth model loaded successfully on {self.device}")

    def preprocess_image(self, frame: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """Preprocess the input frame for depth estimation."""
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

    def pointcloud_to_json(self, pcd: o3d.geometry.PointCloud, material_labels: np.ndarray = None) -> Dict:
        """Convert an Open3D point cloud to a JSON-serializable dictionary with material labels."""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.array([])
        normals = np.asarray(pcd.normals) if pcd.has_normals() else np.array([])
        
        result = {
            'points': points.tolist(),
            'colors': colors.tolist(),
            'normals': normals.tolist()
        }
        
        if material_labels is not None:
            result['material_labels'] = material_labels.tolist()
            
        return result

    def create_point_cloud_with_materials(self, depth: np.ndarray, vis_bgr: np.ndarray, 
                                        material_mask: np.ndarray, focallength_px_tensor: torch.Tensor) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Create a point cloud with material information from the segmentation mask."""
        height, width = depth.shape
        cx, cy = width / 2.0, height / 2.0
        focallength_px = focallength_px_tensor.item()

        y_indices, x_indices = np.indices((height, width), dtype=np.float32)
        z_values = depth.astype(np.float32)
        z_values[z_values < 1e-6] = 1e-6  # Prevent division by zero

        X = (x_indices - cx) * z_values / focallength_px
        Y = (y_indices - cy) * z_values / focallength_px
        Z = z_values

        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        
        # Convert vis_bgr (BGR) to RGB for point cloud colors
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        colors_normalized = vis_rgb.reshape(-1, 3) / 255.0
        
        # Flatten material mask
        material_labels = material_mask.reshape(-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

        if self.voxel_size > 0:
            # Store original indices before downsampling
            original_indices = np.arange(len(points))
            pcd_with_indices = o3d.geometry.PointCloud()
            pcd_with_indices.points = pcd.points
            pcd_with_indices.colors = pcd.colors
            
            # Downsample
            pcd_downsampled, _, indices = pcd.voxel_down_sample_and_trace(
                voxel_size=self.voxel_size, 
                min_bound=pcd.get_min_bound(), 
                max_bound=pcd.get_max_bound()
            )
            
            # Map material labels to downsampled points
            if len(indices) > 0:
                # Take the first index from each voxel for material label
                downsampled_material_labels = material_labels[np.array([idx[0] for idx in indices if len(idx) > 0])]
            else:
                downsampled_material_labels = material_labels
                
            pcd = pcd_downsampled
            material_labels = downsampled_material_labels

        if self.estimate_normals and pcd.has_points():
            if len(pcd.points) >= 30:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=30)
                )
            else:
                print("DEBUG: Not enough points to estimate normals after downsampling.")

        return pcd, material_labels

    def create_point_cloud(self, depth: np.ndarray, vis_bgr: np.ndarray, focallength_px_tensor: torch.Tensor) -> o3d.geometry.PointCloud:
        """Original create_point_cloud method for backward compatibility."""
        pcd, _ = self.create_point_cloud_with_materials(depth, vis_bgr, np.zeros_like(depth), focallength_px_tensor)
        return pcd

    def apply_color(self, label_mask: np.ndarray) -> np.ndarray:
        """Apply the flammability-based color map to the label mask."""
        print(f"DEBUG: apply_color - label_mask shape: {label_mask.shape}")
        print(f"DEBUG: apply_color - label_mask unique values: {np.unique(label_mask)}")
        vis = np.take(self.srgb_colormap, label_mask, axis=0)  # Map labels to RGB
        return vis[..., ::-1]  # Convert RGB to BGR

    def pointcloud_to_fds_geometry(self, pcd: o3d.geometry.PointCloud, material_labels: np.ndarray) -> str:
        """Convert point cloud with material labels to FDS geometry definitions."""
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            return ""
        
        # Get point cloud bounds
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        
        print(f"DEBUG: Point cloud bounds: min={min_bound}, max={max_bound}")
        
        # Group points by material
        dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                 46, 47, 48, 49, 50, 51, 52, 53, 56]
        
        geometry_blocks = []
        material_definitions = []
        
        # Create material definitions
        used_materials = set()
        for label in np.unique(material_labels):
            if 0 <= label < len(dms46):
                material_id = dms46[label]
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
                            mat_def += f"\n      HEAT_OF_COMBUSTION={props['heat_of_combustion']*1000}"  # Convert to J/kg
                            mat_def += f"\n      SOOT_YIELD={props['soot_yield']}"
                            mat_def += f"\n      CO_YIELD={props['co_yield']}"
                        
                        mat_def += " /\n"
                        material_definitions.append(mat_def)
                        
                        # Create surface for material
                        surf_def = f"&SURF ID='{flam_class}_SURF' MATL_ID='{flam_class}' THICKNESS=0.02"
                        if props['ignition_temperature'] is not None:
                            hrr = self.fds_config.get_hrr_per_unit_area(flam_class)
                            if hrr > 0:
                                surf_def += f" HRRPUA={hrr}"
                        surf_def += f" COLOR='{props['color']}' /\n"
                        material_definitions.append(surf_def)

        # Create voxelized geometry
        grid_size = self.fds_config.grid_resolution
        
        # Create 3D grid
        x_bins = np.arange(min_bound[0], max_bound[0] + grid_size, grid_size)
        y_bins = np.arange(min_bound[1], max_bound[1] + grid_size, grid_size)
        z_bins = np.arange(min_bound[2], max_bound[2] + grid_size, grid_size)
        
        # Assign points to grid cells
        x_indices = np.digitize(points[:, 0], x_bins) - 1
        y_indices = np.digitize(points[:, 1], y_bins) - 1
        z_indices = np.digitize(points[:, 2], z_bins) - 1
        
        # Create obstacles for each material type
        for material_class in used_materials:
            # Find points belonging to this material
            material_points = []
            for i, label in enumerate(material_labels):
                if 0 <= label < len(dms46):
                    material_id = dms46[label]
                    if material_id in self.flammability_map and self.flammability_map[material_id] == material_class:
                        material_points.append(i)
            
            if not material_points:
                continue
                
            # Get grid cells for this material
            mat_x_indices = x_indices[material_points]
            mat_y_indices = y_indices[material_points]
            mat_z_indices = z_indices[material_points]
            
            # Group adjacent cells into larger blocks (simple clustering)
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
        """Create complete FDS input file."""
        
        # Get bounds and create domain
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        
        # Expand domain slightly
        domain_padding = 0.5  # meters
        x1, y1, z1 = min_bound - domain_padding
        x2, y2, z2 = max_bound + domain_padding
        
        # Ensure minimum domain size
        if x2 - x1 < 2.0: x2 = x1 + 2.0
        if y2 - y1 < 2.0: y2 = y1 + 2.0
        if z2 - z1 < 2.0: z2 = z1 + 2.0
        
        # Calculate mesh dimensions
        nx = max(20, int((x2 - x1) / self.fds_config.grid_resolution))
        ny = max(20, int((y2 - y1) / self.fds_config.grid_resolution))
        nz = max(20, int((z2 - z1) / self.fds_config.grid_resolution))
        
        fds_input = f"""! FDS Input File Generated by FlameProcessor
! Generated from point cloud with {len(pcd.points)} points

&HEAD CHID='fire_simulation' TITLE='Point Cloud Fire Simulation' /

&MESH IJK={nx},{ny},{nz} XB={x1:.3f},{x2:.3f},{y1:.3f},{y2:.3f},{z1:.3f},{z2:.3f} /

&TIME T_END={self.fds_config.simulation_time} /

&MISC TMPA={self.fds_config.ambient_temperature} /

&DUMP DT_DEVC={self.fds_config.output_interval} DT_HRR={self.fds_config.output_interval} /

! Boundary conditions
&SURF ID='OPEN' RGB=255,255,255 /
&VENT XB={x1:.3f},{x1:.3f},{y1:.3f},{y2:.3f},{z1:.3f},{z2:.3f} SURF_ID='OPEN' /
&VENT XB={x2:.3f},{x2:.3f},{y1:.3f},{y2:.3f},{z1:.3f},{z2:.3f} SURF_ID='OPEN' /
&VENT XB={x1:.3f},{x2:.3f},{y1:.3f},{y1:.3f},{z1:.3f},{z2:.3f} SURF_ID='OPEN' /
&VENT XB={x1:.3f},{x2:.3f},{y2:.3f},{y2:.3f},{z1:.3f},{z2:.3f} SURF_ID='OPEN' /
&VENT XB={x1:.3f},{x2:.3f},{y1:.3f},{y2:.3f},{z2:.3f},{z2:.3f} SURF_ID='OPEN' /

! Ground plane
&OBST XB={x1:.3f},{x2:.3f},{y1:.3f},{y2:.3f},{z1:.3f},{z1+0.01:.3f} SURF_ID='INERT' /

"""
        
        # Add geometry from point cloud
        geometry = self.pointcloud_to_fds_geometry(pcd, material_labels)
        fds_input += geometry
        
        # Add ignition source
        if ignition_point is None:
            # Default ignition at center bottom
            ignition_point = ((x1 + x2) / 2, (y1 + y2) / 2, z1 + 0.1)
        
        ix, iy, iz = ignition_point
        fds_input += f"""
! Ignition source
&SURF ID='IGNITOR' HRRPUA=500 RAMP_Q='IGNITION_RAMP' COLOR='RED' /
&RAMP ID='IGNITION_RAMP' T=0 F=0 /
&RAMP ID='IGNITION_RAMP' T=5 F=1 /
&RAMP ID='IGNITION_RAMP' T=10 F=0 /
&OBST XB={ix-0.1:.3f},{ix+0.1:.3f},{iy-0.1:.3f},{iy+0.1:.3f},{iz:.3f},{iz+0.1:.3f} SURF_ID='IGNITOR' /

! Output devices
&DEVC XYZ={ix:.3f},{iy:.3f},{iz+0.5:.3f} QUANTITY='TEMPERATURE' ID='TEMP_CENTER' /
&DEVC XYZ={ix:.3f},{iy:.3f},{iz+1.0:.3f} QUANTITY='VELOCITY' ID='VEL_CENTER' /

! Slice outputs for visualization
&SLCF PBZ={iz+0.5:.3f} QUANTITY='TEMPERATURE' /
&SLCF PBZ={iz+0.5:.3f} QUANTITY='HRRPUV' /
&SLCF PBY={(y1+y2)/2:.3f} QUANTITY='TEMPERATURE' /

&TAIL /
"""
        
        return fds_input

    def run_fds_simulation(self, fds_input: str) -> Dict:
        """Run FDS simulation and return results."""
        if not self.enable_fds or not self.fds_executable:
            return {'error': 'FDS not available or disabled'}
        
        # Create temporary directory for simulation
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, 'fire_simulation.fds')
            
            # Write FDS input file
            with open(input_file, 'w') as f:
                f.write(fds_input)
            
            print(f"DEBUG: Running FDS simulation in {temp_dir}")
            
            try:
                # Run FDS simulation
                result = subprocess.run([self.fds_executable, input_file], 
                                      cwd=temp_dir, 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=300)  # 5 minute timeout
                
                if result.returncode == 0:
                    print("DEBUG: FDS simulation completed successfully")
                    
                    # Read output files
                    sim_results = {
                        'status': 'success',
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'output_files': [],
                        'hrr_data': [],
                        'temperature_data': []
                    }
                    
                    # Parse HRR file if it exists
                    hrr_file = os.path.join(temp_dir, 'fire_simulation_hrr.csv')
                    if os.path.exists(hrr_file):
                        try:
                            with open(hrr_file, 'r') as f:
                                lines = f.readlines()
                                # Skip header lines
                                data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
                                if len(data_lines) > 1:  # Has data beyond header
                                    sim_results['hrr_data'] = data_lines
                        except Exception as e:
                            print(f"WARNING: Could not read HRR file: {e}")
                    
                    # Parse device output file if it exists
                    devc_file = os.path.join(temp_dir, 'fire_simulation_devc.csv')
                    if os.path.exists(devc_file):
                        try:
                            with open(devc_file, 'r') as f:
                                lines = f.readlines()
                                data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
                                if len(data_lines) > 1:
                                    sim_results['temperature_data'] = data_lines
                        except Exception as e:
                            print(f"WARNING: Could not read device file: {e}")
                    
                    # List all output files
                    for file in os.listdir(temp_dir):
                        if file.startswith('fire_simulation') and file.endswith(('.csv', '.smv', '.s3d')):
                            sim_results['output_files'].append(file)
                    
                    return sim_results
                    
                else:
                    print(f"ERROR: FDS simulation failed with return code {result.returncode}")
                    return {
                        'status': 'error',
                        'return_code': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
                    
            except subprocess.TimeoutExpired:
                print("ERROR: FDS simulation timed out")
                return {'status': 'timeout', 'error': 'Simulation timed out after 5 minutes'}
            except Exception as e:
                print(f"ERROR: Failed to run FDS simulation: {e}")
                return {'status': 'error', 'error': str(e)}

    def create_fire_particles(self, pcd: o3d.geometry.PointCloud, material_labels: np.ndarray,
                            ignition_point: Tuple[float, float, float] = None) -> np.ndarray:
        """Create fire particle positions based on combustible materials."""
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            return np.array([]).reshape(0, 3)
        
        # Find combustible materials
        dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                 46, 47, 48, 49, 50, 51, 52, 53, 56]
        
        combustible_points = []
        
        for i, label in enumerate(material_labels):
            if 0 <= label < len(dms46):
                material_id = dms46[label]
                if material_id in self.flammability_map:
                    flam_class = self.flammability_map[material_id]
                    # Only include combustible materials (not A1)
                    if flam_class != 'A1':
                        combustible_points.append(points[i])
        
        if not combustible_points:
            return np.array([]).reshape(0, 3)
        
        combustible_points = np.array(combustible_points)
        
        # Create fire particles at combustible locations
        # Add some randomness and elevation for visual effect
        fire_particles = []
        
        for point in combustible_points:
            # Create multiple particles per combustible point
            num_particles = np.random.randint(1, 4)  # 1-3 particles per point
            
            for _ in range(num_particles):
                # Add random offset
                offset = np.random.normal(0, 0.05, 3)  # 5cm std dev
                particle_pos = point + offset
                particle_pos[2] += np.random.uniform(0.1, 0.5)  # Elevate particles
                fire_particles.append(particle_pos)
        
        return np.array(fire_particles)

    def process_frame(self, frame: np.ndarray, ignition_point: Tuple[float, float, float] = None) -> Tuple[np.ndarray, Dict]:
        """
        Enhanced process_frame with FDS fire simulation capability.

        Args:
            frame: Input image as a numpy array in BGR format.
            ignition_point: Optional (x, y, z) coordinates for ignition source.

        Returns:
            Tuple containing:
            - processed_frame: Concatenated original and segmented image in BGR.
            - result_dict: Dictionary with flammability classes, point cloud data, and fire simulation results.
        """
        print(f"DEBUG: process_frame - Input frame shape: {frame.shape}")
        h, w = frame.shape[:2]
        print(f"DEBUG: Original dimensions: {h}x{w}")

        # Resize frame (smallest side to 512, maintain aspect ratio)
        scale = 512 / max(h, w)
        new_h = math.ceil(scale * h)
        new_w = math.ceil(scale * w)
        print(f"DEBUG: Resize scale: {scale}, new dimensions: {new_h}x{new_w}")

        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        print(f"DEBUG: Resized frame shape: {resized_frame.shape}")

        # Material Segmentation
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
        print(f"DEBUG: Material input tensor shape: {image.shape}")

        try:
            with torch.no_grad():
                model_output = self.material_model(image)
                prediction = model_output[0].detach().cpu().numpy()[0, 0].astype(np.int64)
                print(f"DEBUG: Material prediction shape: {prediction.shape}")
        except Exception as e:
            print(f"ERROR: Material inference failed: {e}")
            prediction = np.zeros((new_h, new_w), dtype=np.int64)

        vis_bgr = self.apply_color(prediction)
        processed_frame = np.concatenate((resized_frame, vis_bgr), axis=1)

        # Detect flammability classes
        unique_labels = np.unique(prediction)
        dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                 46, 47, 48, 49, 50, 51, 52, 53, 56]
        
        detected_classes = []
        for label in unique_labels:
            if 0 <= label < len(dms46):
                material_id = dms46[label]
                if material_id in self.flammability_map:
                    detected_classes.append(self.flammability_map[material_id])
        detected_classes = sorted(set(detected_classes))
        result_text = "Detected flammability classes: " + ", ".join(detected_classes) if detected_classes else "No materials detected"
        print(result_text)

        # Depth Estimation and Point Cloud Creation
        pil_image, _ = self.preprocess_image(resized_frame)
        try:
            image_tensor = self.depth_transform(pil_image).to(self.device)
            if image_tensor.dtype == torch.float16 and self.device == 'cpu':
                image_tensor = image_tensor.float()

            with torch.no_grad():
                prediction_depth = self.depth_model.infer(image_tensor.unsqueeze(0))
            
            depth_np = prediction_depth["depth"].squeeze().detach().cpu().numpy()
            focallength_px = prediction_depth["focallength_px"].squeeze()

            # Create point cloud with material information
            pcd, material_labels = self.create_point_cloud_with_materials(depth_np, vis_bgr, prediction, focallength_px)
            pointcloud_json = self.pointcloud_to_json(pcd, material_labels)
            
            # Create fire particles
            fire_particles = self.create_fire_particles(pcd, material_labels, ignition_point)
            pointcloud_json['fire_particles'] = fire_particles.tolist()
            
            # FDS Fire Simulation
            fds_results = {'status': 'disabled', 'message': 'FDS simulation disabled'}
            if self.enable_fds and len(pcd.points) > 100:  # Only run if we have enough points
                print("DEBUG: Starting FDS fire simulation...")
                try:
                    fds_input = self.create_fds_input_file(pcd, material_labels, ignition_point)
                    fds_results = self.run_fds_simulation(fds_input)
                    fds_results['input_file'] = fds_input  # Include input for debugging
                    print(f"DEBUG: FDS simulation completed with status: {fds_results.get('status', 'unknown')}")
                except Exception as e:
                    print(f"ERROR: FDS simulation failed: {e}")
                    fds_results = {'status': 'error', 'error': str(e)}
            else:
                if self.enable_fds:
                    fds_results['message'] = f'Insufficient points for simulation (need >100, got {len(pcd.points)})'
            
            pointcloud_json['fds_simulation'] = fds_results
            
        except Exception as e:
            print(f"ERROR: Depth processing failed: {e}")
            pointcloud_json = {
                'error': str(e), 
                'points': [], 
                'colors': [], 
                'normals': [],
                'material_labels': [],
                'fire_particles': [],
                'fds_simulation': {'status': 'error', 'error': 'Depth processing failed'}
            }

        # Add detected classes to result
        pointcloud_json['detected_flammability_classes'] = detected_classes
        pointcloud_json['result_text'] = result_text

        return processed_frame, pointcloud_json
    
    def process_pointcloud(self, point_cloud_data: Dict, ignition_point: Tuple[float, float, float] = None) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Process existing point cloud data with FDS fire simulation.
        
        Args:
            point_cloud_data: Dictionary containing point cloud data with material labels
            ignition_point: Optional ignition source location
            
        Returns:
            Tuple of (enhanced_point_cloud_data, simulation_results)
        """
        if not point_cloud_data or 'points' not in point_cloud_data:
            return None, {"error": "Invalid point cloud data"}
        
        try:
            # Reconstruct point cloud
            points = np.array(point_cloud_data['points'])
            colors = np.array(point_cloud_data.get('colors', []))
            material_labels = np.array(point_cloud_data.get('material_labels', []))
            
            if len(points) == 0:
                return point_cloud_data, {"error": "Empty point cloud"}
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if len(colors) > 0:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # If no material labels, assume all points are highly combustible
            if len(material_labels) == 0:
                material_labels = np.full(len(points), 52)  # D/E class default
                print("WARNING: No material labels found, assuming D/E class for all points")
            
            # Create fire particles
            fire_particles = self.create_fire_particles(pcd, material_labels, ignition_point)
            
            # Enhanced point cloud data
            enhanced_data = point_cloud_data.copy()
            enhanced_data['fire_particles'] = fire_particles.tolist()
            
            # Run FDS simulation if enabled
            if self.enable_fds and len(points) > 100:
                print("DEBUG: Running FDS simulation on existing point cloud...")
                try:
                    fds_input = self.create_fds_input_file(pcd, material_labels, ignition_point)
                    fds_results = self.run_fds_simulation(fds_input)
                    fds_results['input_file'] = fds_input
                    
                    enhanced_data['fds_simulation'] = fds_results
                    return enhanced_data, fds_results
                    
                except Exception as e:
                    error_result = {'status': 'error', 'error': str(e)}
                    enhanced_data['fds_simulation'] = error_result
                    return enhanced_data, error_result
            else:
                message = 'FDS disabled' if not self.enable_fds else f'Insufficient points ({len(points)} < 100)'
                result = {'status': 'skipped', 'message': message}
                enhanced_data['fds_simulation'] = result
                return enhanced_data, result
                
        except Exception as e:
            print(f"ERROR: Point cloud processing failed: {e}")
            return None, {"error": str(e)}

    def get_material_statistics(self, material_labels: np.ndarray) -> Dict:
        """Get statistics about detected materials and their fire properties."""
        dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                 46, 47, 48, 49, 50, 51, 52, 53, 56]
        
        stats = {
            'total_points': len(material_labels),
            'flammability_distribution': {},
            'fire_risk_assessment': {
                'overall_risk': 'LOW',
                'combustible_percentage': 0.0,
                'highly_combustible_percentage': 0.0
            }
        }
        
        flammability_counts = {}
        
        for label in material_labels:
            if 0 <= label < len(dms46):
                material_id = dms46[label]
                if material_id in self.flammability_map:
                    flam_class = self.flammability_map[material_id]
                    flammability_counts[flam_class] = flammability_counts.get(flam_class, 0) + 1
        
        # Calculate percentages
        total_points = len(material_labels)
        if total_points > 0:
            for flam_class, count in flammability_counts.items():
                percentage = (count / total_points) * 100
                stats['flammability_distribution'][flam_class] = {
                    'count': count,
                    'percentage': percentage
                }
        
        # Risk assessment
        combustible_classes = ['A2', 'B/C', 'D/E', 'F']
        highly_combustible_classes = ['D/E', 'F']
        
        combustible_count = sum(flammability_counts.get(cls, 0) for cls in combustible_classes)
        highly_combustible_count = sum(flammability_counts.get(cls, 0) for cls in highly_combustible_classes)
        
        if total_points > 0:
            combustible_pct = (combustible_count / total_points) * 100
            highly_combustible_pct = (highly_combustible_count / total_points) * 100
            
            stats['fire_risk_assessment']['combustible_percentage'] = combustible_pct
            stats['fire_risk_assessment']['highly_combustible_percentage'] = highly_combustible_pct
            
            # Determine overall risk
            if highly_combustible_pct > 50:
                stats['fire_risk_assessment']['overall_risk'] = 'EXTREME'
            elif highly_combustible_pct > 25:
                stats['fire_risk_assessment']['overall_risk'] = 'HIGH'
            elif combustible_pct > 50:
                stats['fire_risk_assessment']['overall_risk'] = 'MODERATE'
            elif combustible_pct > 10:
                stats['fire_risk_assessment']['overall_risk'] = 'LOW'
            else:
                stats['fire_risk_assessment']['overall_risk'] = 'MINIMAL'
        
        return stats


# Initialize processor
processor = FlameProcessor(enable_fds=True)  # Set to False to disable FDS simulation
app = processor.app