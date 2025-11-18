def get_config(args):
    """
    Configure training parameters for specific datasets

    Parameters
    ----------
    args : object
        Configuration object containing model parameters.
        Requires 'dataset', 'train_epochs' and 'valid_epochs' attributes

    Returns
    -------
    object
        Modified configuration object with dataset-specific settings

    Raises
    ------
    NotImplementedError
        If requested dataset is not in supported configurations
    """
    
    if args.dataset == 'MSRCV1':
        args.train_epochs = 200
    elif args.dataset == 'synthetic3d':
        args.train_epochs = 200
    elif args.dataset == 'uci-digit':
        args.train_epochs = 200
    elif args.dataset == 'ALOI':
        args.train_epochs = 200
    elif args.dataset == 'handwritten': # Handwritten的参数用1.0
        args.train_epochs = 200
    elif args.dataset == 'Scene15':
        args.train_epochs = 200
    elif args.dataset == 'Animal':
        args.train_epochs = 100
    # else:
    #     raise NotImplementedError(f"No configuration found for dataset {args.dataset}")
    
    return args