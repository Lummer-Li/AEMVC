import torch
import configs
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import dataset_with_info
from utils import set_seed, get_logger
from models import MyNet, soft_thresholding
from metrics import clusteringMetrics
    
def main(args, logger, datasetforuse, data_size, view_num, nc, input_dims, gnd):
    """Main training pipeline for multi-view clustering model
    
    Args:
        args: Configuration parameters
        logger: Configured logging handler
        datasetforuse: Dataset object containing multi-view data
        data_size: Number of samples
        view_num: Number of data views/modalities
        nc: Number of clusters
        input_dims: List of input dimensions per view
        gnd: Ground truth labels
    
    Returns:
        Tuple of evaluation metrics (ACC, NMI, Purity, ARI, Fscore, Precision, Recall)
    """
    
    # Initialize data loaders with full dataset utilization
    train_loader = DataLoader(
        datasetforuse, 
        batch_size=args.batch_size, 
        shuffle=True,      # Enable random sampling
        drop_last=False    # Use all samples including partial batches
    ) 
    test_loader = DataLoader(
        datasetforuse, 
        batch_size=args.batch_size, 
        shuffle=False,     # Maintain original order for evaluation
        drop_last=False
    )

    # Display configuration header
    print("="*120)
    logger.info(str(args))  # Log all hyperparameters
    print("="*120)

    # Model initialization with multi-view architecture
    model = MyNet(
        args=args,          # Configuration object
        input_dims=input_dims,  # Feature dimensions per view
        view_num=view_num,  # Number of data modalities
        class_num=nc        # Target cluster count
    ).to(args.device)       # Device placement (GPU/CPU)

    # Adam optimizer with L2 regularization
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr,              # Learning rate
        weight_decay=args.weight_decay  # Weight decay coefficient
    )

    # Mean Squared Error loss for reconstruction
    mse_loss_fn = nn.MSELoss()

    # Training metrics storage
    losses = []

    # Main training loop
    for epoch in range(args.train_epochs):
        total_loss = 0.  # Epoch loss accumulator
        
        # Batch processing
        for x, y, idx, inpu, mask in train_loader:
            loss_rec = 0.    # Reconstruction loss
            loss_dec = 0.    # Decomposition loss 
            loss_align = 0.  # Cross-view alignment loss
                        
            # Training mode setup
            model.train()
            
            # View-specific preprocessing
            for v in range(view_num):
                # Apply missing data mask
                x[v] = (x[v] * mask[:, v].unsqueeze(1))
                x[v] = x[v].to(args.device)  # Device transfer
                y = y.to(args.device)

            # Forward pass with feature decomposition
            xrs, zs, Us, Sigmas, Vs = model(x, clustering=False)
            
            # Gradient reset
            optimizer.zero_grad()
            
            # Loss computation per view
            for v in range(view_num):
                # Reconstruction loss (mask-aware)
                loss_rec += mse_loss_fn(
                    xrs[v] * mask[:, v].unsqueeze(1).to(args.device),
                    x[v] * mask[:, v].unsqueeze(1).to(args.device)
                )
                
                # Singular value thresholding
                Sigma_v = soft_thresholding(Sigmas[v], model.thresholds[v])
                
                # Decomposition loss (sparsity promotion)
                loss_dec += ((Sigma_v[torch.where(Sigma_v > 0)] - Sigma_v.max()) ** 2).sum()

                # Low-rank embedding construction
                embedded_v = torch.mm(Us[v], torch.mm(torch.diag(Sigma_v), Vs[v].t()))
                
                # # Alignment regularization
                loss_align += 0.001 * Sigma_v.sum() / view_num        

                # Cross-view consistency
                for w in range(view_num):
                    if w == v: continue
                    Sigma_w = soft_thresholding(Sigmas[w], model.thresholds[w])
                    embedded_w = torch.mm(Us[w], torch.mm(torch.diag(Sigma_w), Vs[w].t()))
                    loss_align += mse_loss_fn(embedded_v, model.projection(embedded_w))
                    
            # Composite loss calculation
            loss = loss_rec + 1 * loss_dec + 2.0 * loss_align / (view_num * (view_num - 1))
            
            # Backpropagation
            total_loss += loss.item() 
            loss.backward()  # Gradient computation
            optimizer.step()                  # Parameter update
            
        # Epoch logging    
        losses.append(total_loss)
        if (epoch + 1) % 10 == 0:
            print(f'epoch: {epoch+1}, total loss: {loss.item():.4f}, rec loss: {loss_rec:.4f}, dec loss: {loss_dec:.4f}, align loss: {loss_align:.4f}')
        
        # Validation phase
        if (epoch + 1) % args.valid_epochs == 0:
            model.eval()  # Inference mode
            with torch.no_grad():  # Disable gradient tracking
                svd_features = []
                
                # Batch processing for evaluation
                for x, y, idx, inpu, mask in test_loader:
                    # View masking
                    for v in range(view_num):
                        x[v] = (x[v] * mask[:, v].unsqueeze(1))
                        x[v] = x[v].to(args.device)
                    
                    # Feature decomposition
                    xrs, zs, Us, Sigmas, Vs = model(x)
                    
                    # Multi-view feature aggregation
                    svd_feature = []
                    for v in range(view_num):
                        Sigma_v = soft_thresholding(Sigmas[v], model.thresholds[v])
                        embedded_v = torch.mm(Us[v], torch.mm(torch.diag(Sigma_v), Vs[v].t()))
                        svd_feature.append(embedded_v)

                    # Concatenated view features
                    svd_feature = torch.concat(svd_feature, dim=1)
                    svd_features.append(svd_feature)

                # Clustering evaluation
                svd_features = torch.cat(svd_features, dim=0)
                y_pred = model.clustering(svd_features, nc).cpu().numpy()
                
                # Metric computation
                ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(gnd, y_pred) 
                
                # Result logging
                info = {
                    "epoch": epoch + 1, 
                    "acc": '%.4f'%ACC, 
                    "nmi": '%.4f'%NMI,
                    "ari": '%.4f'%ARI,
                    "Purity": '%.4f'%Purity,
                    "fscore": '%.4f'%Fscore,
                    "percision": '%.4f'%Precision,
                    "recall": '%.4f'%Recall
                }
                logger.info(str(info))

            # Final epoch return
            if (epoch + 1) == args.train_epochs:
                return ACC, NMI, Purity, ARI, Fscore, Precision, Recall

if __name__ == '__main__':
    """
    Deep Multi-view Clustering with Intra-view Similarity and Cross-view Correlation Learning (MISCC)
    Main execution pipeline for multi-view clustering training
    
    Steps:
    1. Parameter configuration
    2. Dataset preparation
    3. Model initialization
    4. Training model and optimize its parameters
    5. Final evaluation
    """
    # Argument parsing for training configuration
    parser = argparse.ArgumentParser()
    # Experiment setup
    parser.add_argument('--seed', type=int, default=10,
                      help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='ALOI',
                      help='Dataset name from available options')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Number of samples per training batch')
    parser.add_argument('--missing_rate', type=float, default=0.,
                      help='Missing rate of datasets')
    parser.add_argument('--supervision', type=bool, default=False,
                      help='Supervised/Unsupervised Tasks')
    
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                      help='Momentum factor (not used in Adam)')
    parser.add_argument('--weight_decay', type=float, default=0,
                      help='Weight decay (L2 penalty)')
    
    # Model architecture
    parser.add_argument('--embedding_dim', type=int, default=256,
                      help='Dimension of encoder hidden layer')
    parser.add_argument('--hidden_dims', type=list, default=[1024, 1024],
                      help='Dimension of each hidden layer')
    
    # Training schedule
    parser.add_argument('--train_epochs', type=int, default=200,
                      help='Training epochs for reconstruction')
    parser.add_argument('--valid_epochs', type=int, default=50,
                      help='Cluster validation interval during training')
    
    # Advanced configurations
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Computation device: cuda:id or cpu')
    parser.add_argument('--save_flag', type=bool, default=False,
                      help='Flag for model checkpoint saving')
    
    args = parser.parse_args()
    args = configs.get_config(args)

    # Environment setup
    logger = get_logger(__file__, args.dataset)
    datasetforuse, data_size, view_num, nc, input_dims, gnd = dataset_with_info(args.dataset, missing_rate=args.missing_rate)
    
    acc_list, nmi_list, ari_list, pur_list, fscore_list = [], [], [], [], []
    for i in range(5):                  
        print('======================== current seed %d ======================='%(i+1))
        set_seed(i+1)
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = main(args, logger, datasetforuse, data_size, view_num, nc, input_dims, gnd)
        acc_list.append(ACC)
        nmi_list.append(NMI)
        ari_list.append(ARI)
        pur_list.append(Purity)
        fscore_list.append(Fscore)

    print('Final results (Standard and Variance): ')
    print('ACC: ave|{:04f} std|{:04f}'.format(np.mean(acc_list), np.std(acc_list, ddof=1)))
    print('NMI: ave|{:04f} std|{:04f}'.format(np.mean(nmi_list), np.std(nmi_list, ddof=1)))
    print('PUR: ave|{:04f} std|{:04f}'.format(np.mean(pur_list), np.std(pur_list, ddof=1)))
    print('ARI: ave|{:04f} std|{:04f}'.format(np.mean(ari_list), np.std(ari_list, ddof=1)))
    print('Fscore: ave|{:04f} std|{:04f}'.format(np.mean(fscore_list), np.std(fscore_list, ddof=1)))