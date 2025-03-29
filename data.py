"""
@Author: Said Ohamouddou
@File: main.py
@Time: 2025/02/26 13:18 PM
"""

import os
import sys
import numpy as np
import h5py
import torch
from torch_geometric.data import Dataset, Data
import open3d as o3d
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(42)

def load_point_cloud(file_path):
    """Load point cloud data from various formats (xyz, pts, txt)."""
    file_extension = Path(file_path).suffix.lower()
    try:
        if file_extension == '.xyz':
            points = np.loadtxt(file_path)
            return points[:, :3]  # Take only x, y, z coordinates
        elif file_extension == '.pts':
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if not all(c.isdigit() or c == '.' or c == '-' or c.isspace() for c in first_line):
                    points = np.loadtxt(file_path, skiprows=1)
                else:
                    f.seek(0)
                    points = np.loadtxt(file_path)
            return points[:, :3]
        elif file_extension == '.txt':
            points = np.loadtxt(file_path)
            return points[:, :3]
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise

def point_selection(point_cloud_path, target_point_count):
    """Downsample point cloud to target point count using farthest point sampling."""
    try:
        points = load_point_cloud(point_cloud_path)
        if len(points) < target_point_count:
            logger.warning(f"Point cloud {point_cloud_path} has fewer points ({len(points)}) than target ({target_point_count})")
            points = np.repeat(points, (target_point_count // len(points)) + 1, axis=0)[:target_point_count]
            return points

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        pcd_down = point_cloud.farthest_point_down_sample(target_point_count)
        return np.asarray(pcd_down.points)
    except Exception as e:
        logger.error(f"Error in point selection for {point_cloud_path}: {str(e)}")
        raise
# Data augmentation functions
def translate_pointcloud(pointcloud):
    """Apply random translation augmentation."""
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def normalize_pc(points):
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
    points /= furthest_distance
    return points
def load_data(num_points):
    """Load and process point cloud data, saving to H5 format."""
    folder_path = "data"
    h5_file = "point_cloud_data.h5"
    h5_path = os.path.join(folder_path, h5_file)
    
    try:
        classes = sorted([d for d in os.listdir(folder_path) 
                        if os.path.isdir(os.path.join(folder_path, d))])
        
        if not classes:
            raise ValueError(f"No class directories found in {folder_path}")

        point_clouds = []
        labels = []
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(folder_path, class_name)
            files = [f for f in os.listdir(class_path) 
                    if f.endswith(('.xyz', '.pts', '.txt'))]
            
            if not files:
                logger.warning(f"No valid files found in class {class_name}")
                continue
                
            for file_name in files:
                try:
                    file_path = os.path.join(class_path, file_name)
                    points = point_selection(file_path, num_points)
                    point_clouds.append(points)
                    labels.append(np.array([class_idx]))
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue

        if not point_clouds:
            raise ValueError("No valid point cloud data was loaded")

        point_clouds = np.array(point_clouds)
        labels = np.array(labels)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('point_clouds', data=point_clouds)
            f.create_dataset('labels', data=labels)
            f.create_dataset('classes', data=np.array(classes, dtype='S'))

        logger.info("Data loading completed successfully")
        return h5_path
    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}")
        raise

def data_split(h5_path, test_ratio=0.2):
    """Split data into train and test sets."""
    try:
        with h5py.File(h5_path, 'r') as f:
            point_clouds = f['point_clouds'][:]
            labels = f['labels'][:]
            classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]

        train_point_clouds, test_point_clouds, train_labels, test_labels = train_test_split(
            point_clouds, labels, test_size=test_ratio, stratify=labels, random_state=42
        )

        with h5py.File('data/data_split.h5', 'w') as f:
            train_group = f.create_group('train')
            train_group.create_dataset('point_clouds', data=train_point_clouds)
            train_group.create_dataset('labels', data=train_labels)
            
            test_group = f.create_group('test')
            test_group.create_dataset('point_clouds', data=test_point_clouds)
            test_group.create_dataset('labels', data=test_labels)
            
            f.create_dataset('classes', data=np.array(classes, dtype='S'))

        logger.info("Data splitting completed successfully")
    except Exception as e:
        logger.error(f"Error in data_split: {str(e)}")
        raise

class TreePointCloudDataset(Dataset):
    def __init__(self, num_points, partition='train', transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.num_points = num_points
        self.partition = partition
        
        try:
            if not os.path.exists('data/data_split.h5'):
                h5_path = load_data(self.num_points)
                data_split(h5_path)

            with h5py.File('data/data_split.h5', 'r') as f:
                self.data = f[partition]['point_clouds'][:]
                self.label = f[partition]['labels'][:]
                self.classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise

    def len(self):
        return len(self.data)

    def get(self, idx):
        try:
            pointcloud = self.data[idx][:self.num_points].copy()
            label = self.label[idx]
            
            # Apply augmentation only for training data
            if self.partition == 'train':
                pointcloud = translate_pointcloud(pointcloud)
                np.random.shuffle(pointcloud)
            
            # Normalize the point cloud
            pointcloud = normalize_pc(pointcloud)
            
            # Convert to PyG Data object
            data = Data(
                pos=torch.from_numpy(pointcloud).float(),  # Node positions (N, 3)
                y=torch.from_numpy(label).long(),  # Labels
                num_nodes=self.num_points
            )
            
            # Calculate edge index using k-nearest neighbors
            if self.partition == 'train':
                data = self._add_edges(data)
            
            return data
        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            raise

    def _add_edges(self, data, k=6):
        """Add edge connectivity using k-nearest neighbors."""
        from torch_geometric.nn import knn_graph
        
        # Create edge index using k-nearest neighbors
        edge_index = knn_graph(data.pos, k=k, loop=False)
        data.edge_index = edge_index
        
        return data

def analyze_class_distribution(dataset, title="Class Distribution"):
    """Analyze and display class distribution in a dataset."""
    class_counts = {}
    total_samples = len(dataset)
    
    for i in range(len(dataset)):
        label = dataset.label[i].item()
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    logger.info(f"\n{title}:")
    logger.info("-" * 50)
    logger.info(f"{'Class':<20} {'Count':<10} {'Percentage':<10}")
    logger.info("-" * 50)
    
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"{class_name:<20} {count:<10} {percentage:>6.2f}%")
    
    logger.info("-" * 50)
    logger.info(f"Total samples: {total_samples}\n")
    
    return class_counts

if __name__ == '__main__':
    try:
        # Create datasets
        train_dataset = TreePointCloudDataset(1024, 'train')
        test_dataset = TreePointCloudDataset(1024, 'test')
        print(test_dataset.classes)
        print(train_dataset.classes)
        # Analyze distributions
        train_dist = analyze_class_distribution(train_dataset, "Training Set Distribution")
        test_dist = analyze_class_distribution(test_dataset, "Test Set Distribution")
        
        # Verify data loading
        sample_data = train_dataset[0]
        logger.info(f"\nSample data attributes:")
        logger.info(f"Point cloud shape: {sample_data.pos.shape}")
        logger.info(f"Edge index shape: {sample_data.edge_index.shape}")
        logger.info(f"Label: {sample_data.y}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)
