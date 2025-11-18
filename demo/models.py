import torch
import torch.nn as nn
import torch_clustering
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Flexible multi-layer perceptron encoder with optional batch normalization
    
    Args:
        dims (list): Layer dimensions [input_dim, hidden1, ..., output_dim]
        bn (bool): Add batch norm after hidden layers
    """
    def __init__(self, dims, bn = False):
        super(Encoder, self).__init__()
        assert len(dims) >= 2
        models = []

        # Construct Hidden Layer
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))      # Add Linear Layer
            if i != len(dims) - 2:
                models.append(nn.ReLU(inplace=True))            # Add activate function

        self.models = nn.Sequential(*models)

    def forward(self, X):
        """Input shape: (batch_size, input_dim)"""
        return self.models(X)
    
class Decoder(nn.Module):
    """
    Multi-layer perceptron decoder for feature reconstruction
    
    Architecture: Series of linear layers with ReLU activation on final layer
    Typical use: Expanding latent representations to high-dimensional outputs
    """
    def __init__(self, dims):
        """
        Initialize decoder layers
        
        Args:
            dims (list): Layer dimension sequence. Example:
                [256, 512, 784] creates:
                - Input layer: 256 units
                - Hidden layer: 512 units 
                - Output layer: 784 units
        """
        super(Decoder, self).__init__()
        
        # Layer container initialization
        models = []
        
        # Layer construction loop (iterates for N-1 layers where N = len(dims))
        for i in range(len(dims) - 1):
            # Linear transformation: dims[i] → dims[i+1]
            models.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add ReLU only after final linear layer
            # Condition: i == len(dims)-2 indicates last layer index
            if i == len(dims) - 2:
                models.append(nn.ReLU())
        
        # Sequential container for layer execution
        self.models = nn.Sequential(*models)
    
    def forward(self, X):
        """
        Forward pass processing
        
        Args:
            X (Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor: Reconstructed output of shape (batch_size, output_dim)
        """
        # Sequential processing through defined layers
        return self.models(X)


def soft_thresholding(singular_values, threshold):
    """
    Apply soft thresholding operator for sparse regularization
    
    Implements the element-wise operation:
        output = sign(x) * max(|x| - threshold, 0)
    
    Args:
        singular_values (torch.Tensor): Input tensor of singular values
        threshold (float/torch.Tensor): Non-negative shrinkage threshold. 
            Values should be >=0 for proper sparsification.
    
    Returns:
        torch.Tensor: Thresholded values with same shape as input
    
    Example:
        >>> x = torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.5])
        >>> soft_thresholding(x, 0.7)
        tensor([-1.3000,  0.0000,  0.0000,  0.0000,  0.8000])
    
    Applications:
        - Sparse coding
        - Low-rank matrix recovery
        - Signal denoising
        - L1-norm regularization
    
    Note:
        For numerical stability, ensure threshold is non-negative.
        Negative thresholds will behave like absolute value shrinkage.
    """
    return torch.sign(singular_values) * torch.clamp(torch.abs(singular_values) - threshold, min=0)

class MyNet(nn.Module):
    """
    Multi-view autoencoder network with feature decomposition
    Key Components:
        - View-specific encoders/decoders
        - Latent space projection
        - Singular Value Decomposition (SVD) regularization
    """
    def __init__(self, args, input_dims, view_num, class_num):
        """
        Args:
            args: Configuration parameters
            input_dims (list): Input dimensions for each view
            view_num: Number of data views/modalities
            class_num: Number of target classes
        """
        super().__init__()
        # Initialize architecture parameters
        self.input_dims = input_dims  # List of input dimensions per view
        self.view = view_num          # Number of data views/modalities
        self.class_num = class_num    # Number of target classes
        self.embedding_dim = args.embedding_dim  # Latent space dimension
        self.h_dims = args.hidden_dims  # Encoder hidden layer dimensions
        self.device = args.device     # Computation device

        # Reverse hidden dims for decoder construction
        h_dims_reverse = list(reversed(args.hidden_dims))

        # View-specific components
        self.encoders = []  # Encoder networks for each view
        self.decoders = []  # Decoder networks for each view
        self.thresholds = []  # Learnable thresholds per view
        for v in range(self.view):
            # Encoder architecture: Input -> Hidden Layers -> Embedding
            self.encoders.append(Encoder([input_dims[v]] + self.h_dims + [self.embedding_dim], bn=True).to(self.device))
            # Decoder architecture: Embedding -> Reversed Hidden -> Input
            self.decoders.append(Decoder([self.embedding_dim] + h_dims_reverse + [input_dims[v]]).to(args.device))
            # Learnable threshold parameter for each view
            self.thresholds.append(nn.Parameter(torch.tensor(0.)).to(self.device))

        # Register components as proper module lists
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.thresholds = nn.ParameterList(self.thresholds)

        # Shared projection network
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, 2048),  # Expansion layer
            nn.ReLU(),                            # Non-linearity
            nn.Linear(2048, self.embedding_dim),   # Projection layer
            # nn.Softmax()
        )

    def forward(self, xs, clustering=False, target=None):
        """
        Forward pass with view-specific processing
        Args:
            xs: List of input tensors for each view
            clustering: Flag for clustering mode
            target: Optional target labels
            
        Returns:
            xrs: Reconstructed inputs per view
            zs: Latent embeddings per view
            Us: Left singular vectors from SVD
            Sigmas: Singular values
            Vs: Right singular vectors
        """
        xrs = []  # Reconstructed inputs
        zs = []   # Latent embeddings
        Us = []   # SVD components
        Vs = []
        Sigmas = []
        
        for v in range(self.view):
            # Process each view independently
            x = xs[v]
            
            # Encoder forward pass
            z = self.encoders[v](x)
            
            # Decoder reconstruction
            xr = self.decoders[v](z)
            
            # Store outputs
            xrs.append(xr)
            zs.append(z)
            
            # Regularized SVD decomposition
            epsilon = 1e-6  # Numerical stability term
            U, Sigma, V = torch.svd(z + epsilon * torch.randn_like(z).to(z.device))
            
            # Store decomposition results
            Us.append(U)
            Sigmas.append(Sigma)
            Vs.append(V)

        return xrs, zs, Us, Sigmas, Vs

    
    def clustering(self, features, num_clusters):
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': num_clusters,
            'verbose': False
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=10, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features.to(dtype=torch.float64))
        
        return psedo_labels