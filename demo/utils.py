import os
import torch
import random
import logging
import numpy as np

def get_logger(file_name, data_name):
    """
    Create and configure a logger with both file and console outputs
    
    Args:
        file_name (str): Name identifier for the logger (typically __name__)
        data_name (str): Base name for the log file (appended with .log)
        
    Returns:
        logging.Logger: Configured logger instance with:
            - File handler writing to ./logs/{data_name}.log
            - Console stream handler
            - Standard log format
    """
    # Initialize logger with specified name
    logger = logging.getLogger(file_name)
    # Set logging threshold to INFO level
    logger.setLevel(logging.INFO)
    
    # Configure log file path and handler
    filename = "./logs/" + data_name + ".log"  # Log directory: ./logs/
    # filename = data_name + ".log"  # Alternate location (commented out)
    
    # Create file handler with INFO level
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    
    # Define log message format:
    # Timestamp - Logger Name - Log Level - Message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Create console output handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)  # Use same format as file
    
    # Attach handlers to logger
    logger.addHandler(handler)
    logger.addHandler(console)
    
    return logger

def set_seed(seed=42):
    """Initialize all random number generators with specified seed
    
    Ensures reproducibility across:
    - NumPy computations
    - PyTorch CPU/CUDA operations
    - Python built-in random module
    - Hash-based operations
    
    Args:
        seed (int): Random seed value (default=42)
    
    Note:
        Setting `cudnn.deterministic=True` may impact performance
        but is necessary for reproducible GPU results
    """
    # NumPy random number generation
    np.random.seed(seed)
    
    # PyTorch CPU random states
    torch.manual_seed(seed)
    
    # PyTorch CUDA random states
    torch.cuda.manual_seed(seed)          # Current GPU
    torch.cuda.manual_seed_all(seed)      # All GPUs (multi-GPU setups)
    
    # CUDA convolution optimization settings
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner
    torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
    
    # Python built-in random module
    random.seed(seed)
    
    # Environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)  # Prevent hash randomization

def save_model(state, dataset_name):
    """
    Save model state dictionary to specified path with dataset-based naming
    
    Parameters
    ----------
    state : dict
        Model state dictionary containing:
        - 'model_state_dict': Model parameters
        - 'optimizer_state_dict': Optimizer state (optional)
        - 'epoch': Training epoch (optional)
    dataset_name : str
        Identifier for model versioning, used in filename
    
    Raises
    ------
    PermissionError
        If lacking write permissions for model directory
    IOError
        If disk space insufficient or path invalid
    
    Notes
    -----
    - Automatically creates './models' directory if non-existent
    - Uses PyTorch's serialization format (.pth)
    - Recommended to include training metadata in state
    """
    # Create model directory if not exists
    if not os.path.exists('./models'):
        os.makedirs('./models')  # Recursive directory creation
    
    # Construct filesystem path
    save_path = os.path.join('./models', f'{dataset_name}.pth')
    
    # Serialize model state
    torch.save(state, save_path)  # Uses pickle protocol
    
    # User feedback
    print(f'Model checkpoint saved at: {save_path}')