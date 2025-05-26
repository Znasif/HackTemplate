import os
import sys
import torch
import numpy as np
from typing import Dict, Tuple
from .base_processor import BaseProcessor
import open3d as o3d
from plyfile import PlyData
import argparse
import sklearn.cluster

# Add partfield directory to sys.path
partfield_dir = r'/home/znasif/PartField'  # Replace with actual path to partfield directory
if partfield_dir not in sys.path:
    sys.path.insert(0, partfield_dir)

from partfield.config import default_argument_parser, setup
from partfield.model_trainer_pvcnn_only_demo import Model
from lightning.pytorch import Trainer, seed_everything

class PartsProcessor(BaseProcessor):
    def __init__(self,
                 checkpoint_path: str = "/home/znasif/PartField/model/model_objaverse.ckpt",
                 output_dir: str = "test_results",
                 max_num_clusters: int = 18,
                 use_gpu: bool = True,
                 voxel_size: float = 0.01,
                 seed: int = 42):
        """
        Initialize PartsProcessor for part-wise segmentation of point clouds.

        Args:
            checkpoint_path (str): Path to the partfield model checkpoint.
            output_dir (str): Directory to save temporary files and outputs.
            max_num_clusters (int): Maximum number of clusters for segmentation.
            use_gpu (bool): Whether to use GPU acceleration.
            voxel_size (float): Voxel size for point cloud downsampling.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.max_num_clusters = max_num_clusters
        self.voxel_size = voxel_size
        self.seed = seed
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "ply"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "cluster_out"), exist_ok=True)

        # Load partfield model
        self._load_model()

    def _load_model(self):
        """
        Load the partfield model and configuration.
        """
        # Simulate command-line arguments for partfield
        parser = default_argument_parser()
        args = parser.parse_args([
            '--output_dir', self.output_dir,
            '--continue_ckpt', self.checkpoint_path,
            '--seed', str(self.seed),
            '--training_epochs', '1',  # Not used in prediction
            '--save_every_epoch', '1'
        ])
        self.cfg = setup(args, freeze=False)
        self.model = Model(self.cfg)
        self.model.to(self.device)
        self.model.eval()
        print(f"Partfield model loaded successfully on {self.device}")

    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, str]:
        """
        Preprocess point cloud and save as PLY file.

        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud.

        Returns:
            Tuple[np.ndarray, str]: Preprocessed points (N, 3) and path to saved PLY file.
        """
        # Downsample point cloud
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Extract points
        points = np.asarray(pcd.points)
        
        # Generate unique identifiers
        uid = str(hash(str(points.tobytes())) % 1000000)  # Simple hash-based UID
        view_id = "0"
        
        # Save to PLY file
        ply_path = os.path.join(self.output_dir, f"input_{uid}_{view_id}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        
        return points, ply_path

    def load_ply_to_numpy(self, filename: str) -> np.ndarray:
        """
        Load a PLY file and extract points as a (N, 3) NumPy array.

        Args:
            filename (str): Path to the PLY file.

        Returns:
            np.ndarray: Point cloud array of shape (N, 3).
        """
        ply_data = PlyData.read(filename)
        vertex_data = ply_data["vertex"]
        points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
        return points

    def export_pointcloud_with_labels_to_ply(self, points: np.ndarray, labels: np.ndarray, filename: str):
        """
        Export a labeled point cloud to a PLY file with vertex colors.

        Args:
            points (np.ndarray): (N, 3) array of XYZ coordinates.
            labels (np.ndarray): (N,) array of integer labels.
            filename (str): Output PLY file name.
        """
        assert points.shape[0] == labels.shape[0], "Number of points and labels must match"

        # Generate unique colors for each label
        unique_labels = np.unique(labels)
        colormap = plt.cm.get_cmap("tab20", len(unique_labels))
        label_to_color = {label: colormap(i)[:3] for i, label in enumerate(unique_labels)}

        labels = np.squeeze(labels)
        colors = np.array([label_to_color[label] for label in labels])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Point cloud saved to {filename}")

    def predict_features(self, ply_path: str, uid: str, view_id: str) -> np.ndarray:
        """
        Run partfield inference to generate per-point features.

        Args:
            ply_path (str): Path to the input PLY file.
            uid (str): Unique identifier for the model.
            view_id (str): View identifier.

        Returns:
            np.ndarray: Per-point features.
        """
        seed_everything(self.cfg.seed)
        torch.manual_seed(0)
        np.random.seed(0)

        trainer = Trainer(
            devices=-1,
            accelerator="gpu" if self.device == "cuda" else "cpu",
            precision="16-mixed",
            strategy="ddp" if self.device == "cuda" else "auto",
            max_epochs=1,
            log_every_n_steps=1,
            limit_train_batches=3500,
            limit_val_batches=None
        )

        trainer.predict(self.model, ckpt_path=self.cfg.continue_ckpt)
        
        # Load generated features
        feat_path = os.path.join(self.output_dir, f"part_feat_{uid}_{view_id}.npy")
        if not os.path.exists(feat_path):
            feat_path = os.path.join(self.output_dir, f"part_feat_{uid}_{view_id}_batch.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
        
        point_feat = np.load(feat_path)
        point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
        return point_feat

    def solve_clustering(self, points: np.ndarray, point_feat: np.ndarray, uid: str, view_id: str) -> np.ndarray:
        """
        Perform KMeans clustering on point features.

        Args:
            points (np.ndarray): Point cloud coordinates (N, 3).
            point_feat (np.ndarray): Per-point features (N, D).
            uid (str): Unique identifier for the model.
            view_id (str): View identifier.

        Returns:
            np.ndarray: Array of clustering results for different numbers of clusters.
        """
        all_labels = []
        for num_cluster in range(2, self.max_num_clusters + 1):
            clustering = sklearn.cluster.KMeans(n_clusters=num_cluster, random_state=0).fit(point_feat)
            labels = clustering.labels_
            relabel = np.zeros((len(labels), 1))
            for i, label in enumerate(np.unique(labels)):
                relabel[labels == label] = i
            
            # Save clustering results
            fname_clustering = os.path.join(self.output_dir, "cluster_out", f"{uid}_{view_id}_{str(num_cluster).zfill(2)}")
            np.save(fname_clustering, relabel)
            
            # Export colored point cloud
            fname_ply = os.path.join(self.output_dir, "ply", f"{uid}_{view_id}_{str(num_cluster).zfill(2)}.ply")
            self.export_pointcloud_with_labels_to_ply(points, relabel, fname_ply)
            
            all_labels.append(relabel)
        
        return np.array(all_labels)

    def labels_to_json(self, points: np.ndarray, all_labels: np.ndarray) -> Dict:
        """
        Convert clustering labels to JSON format.

        Args:
            points (np.ndarray): Point cloud coordinates (N, 3).
            all_labels (np.ndarray): Array of labels for different cluster counts (K, N, 1).

        Returns:
            Dict: JSON-serializable dictionary containing segmentation data.
        """
        segmentations = []
        for i, labels in enumerate(all_labels):
            segmentations.append({
                'num_clusters': i + 2,
                'labels': labels.squeeze().tolist()
            })
        
        return {
            'points': points.tolist(),
            'segmentations': segmentations,
            'timestamp': np.datetime64('now').astype(str)
        }

    def process_point_cloud(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """
        Process point cloud to generate part-wise segmentation labels.

        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud.

        Returns:
            Tuple[o3d.geometry.PointCloud, Dict]: Original point cloud and JSON with segmentation data.
        """
        try:
            # Preprocess point cloud
            points, ply_path = self.preprocess_point_cloud(pcd)
            
            # Generate unique identifiers
            uid = str(hash(str(points.tobytes())) % 1000000)
            view_id = "0"
            
            # Run partfield inference
            point_feat = self.predict_features(ply_path, uid, view_id)
            
            # Perform clustering
            all_labels = self.solve_clustering(points, point_feat, uid, view_id)
            
            # Convert to JSON
            json_data = self.labels_to_json(points, all_labels)
            
            return pcd, json_data
        
        except Exception as e:
            print(f"Processing error: {e}")
            return pcd, {'error': str(e), 'points': [], 'segmentations': []}

processor = PartsProcessor()
app = processor.app