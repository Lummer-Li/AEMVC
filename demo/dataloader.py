import scipy
import torch 
import random
import sklearn
import numpy as np
from numpy.random import randint
from torch.utils.data import Dataset
from typing import Tuple, List
from sklearn.preprocessing import OneHotEncoder

def get_mask(view_num, data_len, missing_rate):
    """
    To randomly generate missing matrix, simulating incomplete view data with complete view data.

    Parameters
    ----------
    view_num: number of view
    data_len: number of samples
    missing_rate: default(0.), super parameter to control the proportion of missing data
        the larger the value, the more missing data

    Returns
    -------
    missing_matrix: the missing matrix
    """
    mask_seed = 0
    np.random.seed(mask_seed)
    random.seed(mask_seed) 

    missing_rate = missing_rate / view_num              # 将缺失率平均分配到每个视图上，表示每个视图上的平均缺失比例
    one_rate = 1.0 - missing_rate                       # 每个视图中完整数据所占的比例
    if one_rate <= (1 / view_num):                      # 若完整率小于等于单个视图的比例 1/V，则随机选择一个视图为完整视图，并创建掩码矩阵
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:                                   # 如果完整率为 1
        matrix = np.ones((data_len, view_num)).astype(np.int64)          # 创建全1矩阵，表示所有数据完整
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int64)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int64))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int64)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int64)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)
    return matrix

path = '../../datasets/'
def loadData(data_name):
    """
    Load multi-view dataset from .mat file with consistent data structure
    
    Parameters:
        data_name (str): Path to .mat file containing dataset
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - features: NumPy object array of shape (1, n_views) containing view data
            - ground_truth: Flattened integer array of shape (n_samples,)
            
    Raises:
        ValueError: If dataset format is not recognized or required fields are missing
    """
    dataset_config = {
        'synthetic3d': {
            'feature_base': ('X', 3),
            'n_views': 3,
            'label_key': 'Y',
            'transpose': True,
            'label_process': ['squeeze']
        },
        'uci-digit': {
            'feature_keys': ['mfeat_fac', 'mfeat_fou', 'mfeat_kar'],
            'n_views': 3,
            'label_key': 'truth',
            'label_process': ['squeeze']
        },
        'MSRCV1': {
            'feature_base': ('X', 6),
            'n_views': 6,
            'label_key': 'Y',
            'label_process': ['reshape', (210,)]
        },
        'Animal': {
            'feature_base': ('X', 4),
            'n_views': 4,
            'label_key': 'Y',
            'label_process': ['squeeze']
        },
        'Scene15': {
            'feature_base': ('X', 3),
            'n_views': 3,
            'label_key': 'Y',
            'transpose': True,
            'label_process': ['squeeze']
        },
        'handwritten': {
            'feature_base': ('X', 6),
            'n_views': 6,
            'label_key': 'Y',
            'transpose': True,
            'label_process': ['squeeze']
        },
        'ALOI': {
            'feature_base': ('X', 4),
            'n_views': 4,
            'label_key': 'Y',
            'label_process': ['squeeze']
        },
    }

    # Identify dataset type
    dataset_type = next((k for k in dataset_config if k in data_name), None)
    if not dataset_type:
        raise ValueError(f"Unsupported dataset format: {data_name}")

    config = dataset_config[dataset_type]
    data = scipy.io.loadmat(data_name)
    
    # Feature extraction
    features = np.empty((1, config['n_views']), dtype=object)
    
    if 'feature_base' in config:
        base_key, n_views = config['feature_base']
        for i in range(n_views):
            if config.get('transpose', False):
                features[0][i] = data[base_key][i][0].astype(np.float32)
            else:
                features[0][i] = data[base_key][0][i].astype(np.float32)
    elif 'feature_keys' in config:
        for i, key in enumerate(config['feature_keys']):
            features[0][i] = data[key].astype(np.float32)
    else:
        features[0][0] = data[config['features_key']].astype(np.float32)
        if config.get('transpose', False):
            features[0][0] = features[0][0].T

    # Ground truth processing
    if config['label_key'] not in data:
        raise ValueError(f"Missing ground truth key: {config['label_key']}")

    gnd = data[config['label_key']].astype(np.int32)
    
    for operation in config['label_process']:
        if operation[0] == 'squeeze':
            gnd = np.squeeze(gnd)
        elif operation[0] == 'reshape':
            gnd = gnd.reshape(operation[1])

    return features, gnd.flatten()

class MultiViewDataset(Dataset):
    def __init__(self, dataname: str, missing_rate: float):
        """
        Multi-view dataset loader with missing view handling
        
        Args:
            dataname: Name/path of the dataset (without .mat extension)
            missing_rate: Proportion of missing views (0-1)
        """
        # Load and preprocess data
        features, gnd = loadData(path + dataname + '.mat')
        self.num_views = features.shape[1]
        self.num_samples = len(gnd)
        
        # Convert features to tensors and normalize
        self.features = [
            torch.from_numpy(
                sklearn.preprocessing.MinMaxScaler().fit_transform(view)
            ).float()
            for view in features[0]
        ]
        
        # Store ground truth and indices
        self.gnd = torch.from_numpy(gnd).long()
        self.indices = torch.arange(self.num_samples)
        
        # Generate missing view mask
        self.mask = torch.from_numpy(
            get_mask(self.num_views, self.num_samples, missing_rate)
        ).float()
        
        # Validate dimensions
        self._validate_shapes()

    def _validate_shapes(self):
        """Ensure all components have consistent dimensions"""
        assert all(v.shape[0] == self.num_samples for v in self.features), \
            "Feature dimension mismatch"
        assert self.mask.shape == (self.num_samples, self.num_views), \
            "Mask shape mismatch"
        assert self.gnd.shape == (self.num_samples,), \
            "Ground truth shape mismatch"

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            tuple: Contains
                - views: List of tensors (one per view)
                - label: Ground truth label
                - idx: Sample index
                - mask: View availability mask
        """
        return (
            [view[idx] for view in self.features],  # Views
            self.gnd[idx],                          # Label
            torch.from_numpy(np.array(idx)),        # Idx
            self.indices[idx],                      # Index
            self.mask[idx]                          # Availability mask
        )

    @property
    def view_dims(self) -> list:
        """Get dimensionality of each view"""
        return [v.shape[1] for v in self.features]

def dataset_with_info(dataname: str, missing_rate: float = 0.0) -> Tuple[
    MultiViewDataset, int, int, int, List[int], np.ndarray
]:
    """
    Loads a multi-view dataset and provides comprehensive metadata
    
    Args:
        dataname: The name of Dataset file
        missing_rate: Proportion of missing views (0-1)
    
    Returns:
        Tuple containing:
        - Initialized MultiViewDataset
        - Sample count
        - Number of views
        - Cluster count
        - Feature dimensions per view
        - Ground truth labels
    
    Raises:
        ValueError: For invalid inputs or data loading failures
    """
    # Load and validate data
    try:
        features, gnd = loadData(path + dataname + '.mat')
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}") from e

    if features.size == 0:
        raise ValueError("Loaded features array is empty")
    if len(gnd) == 0:
        raise ValueError("No ground truth labels found")

    # Extract dataset characteristics
    num_views = features.shape[1]
    sample_count = features[0][0].shape[0]
    cluster_count = len(np.unique(gnd))
    feature_dims = [features[0][v].shape[1] for v in range(num_views)]

    # Initialize dataset
    dataset = MultiViewDataset(dataname, missing_rate=missing_rate)

    # Display dataset summary
    summary = (
        f"Dataset: {dataname}\n"
        f"Samples: {sample_count:,}\n"
        f"Views: {num_views}\n"
        f"Clusters: {cluster_count}\n"
        f"Feature Dimensions: {feature_dims}"
    )
    print(summary)

    return dataset, sample_count, num_views, cluster_count, feature_dims, gnd
