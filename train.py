import torchvision  # Used for creating image grids for visualizing samples in TensorBoard
from utils.func import *  # Loads helper functions like printing, logging, saving weights, etc.
from modules.loss import *  # Custom loss classes (e.g., KappaLoss, FocalLoss)
from modules.scheduler import *  # Custom learning rate schedulers (e.g., Warmup, Clipped Cosine)


def train(cfg, model, train_dataset, val_dataset, estimator, logger=None):
    device = cfg.base.device  # Get training device from config (usually 'cpu' or 'cuda')
    optimizer = initialize_optimizer(cfg, model)  # Build optimizer (e.g., Adam, SGD)
    train_sampler, val_sampler = initialize_sampler(cfg, train_dataset, val_dataset)  # Sampling strategy (e.g., class-balanced)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(cfg, optimizer)  # Learning rate scheduler and optional warmup
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)  # Loss function setup, may include class weights
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset, train_sampler, val_sampler)  # PyTorch DataLoaders

    model.train()  # Set model to training mode (activates dropout, batchnorm, etc.)
    avg_loss = 0  # Average loss across an epoch
    max_indicator = 0  # Track the best validation score to save the best model

    for epoch in range(1, cfg.train.epochs + 1):  # Train for the total number of epochs
        if train_sampler:
            train_sampler.step()  # Update sampling weights if using progressive balancing

        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()  # Get updated loss weights
            loss_function.weight = weight.to(device)  # Apply updated weights to loss function

        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()  # Adjust learning rate during warmup phase

        epoch_loss = 0  # Track cumulative loss for current epoch
        estimator.reset()  # Reset metric tracking for new epoch
        progress = tqdm(enumerate(train_loader), total=len(train_loader)) if cfg.base.progress else enumerate(train_loader)  # Enable progress bar if set

        for step, train_data in progress:  # Loop through batches
            X, y = train_data  # Unpack input and labels
            X = X.to(device)  # Move input to device
            y = y.to(device)  # Move labels to device
            y = select_target_type(y, cfg.train.criterion)  # Match label type to loss function

            y_pred = model(X)  # Forward pass to get predictions
            loss = loss_function(y_pred, y)  # Compute loss

            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            epoch_loss += loss.item()  # Accumulate loss
            avg_loss = epoch_loss / (step + 1)  # Update average loss

            estimator.update(y_pred, y)  # Update metric tracker
            message = 'epoch: [{} / {}], loss: {:.6f}'.format(epoch, cfg.train.epochs, avg_loss)  # Create message
            if cfg.base.progress:
                progress.set_description(message)  # Update tqdm description

        if not cfg.base.progress:
            print(message)  # Print progress if tqdm not used

        train_scores = estimator.get_scores(4)  # Collect metrics
        scores_txt = ', '.join(['{}: {}'.format(metric, score) for metric, score in train_scores.items()])  # Format metrics
        print('Training metrics:', scores_txt)  # Display metrics

        curr_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
        if logger:
            for metric, score in train_scores.items():
                logger.add_scalar('training {}'.format(metric), score, epoch)  # Log each metric
            logger.add_scalar('training loss', avg_loss, epoch)  # Log loss
            logger.add_scalar('learning rate', curr_lr, epoch)  # Log learning rate

        if cfg.train.sample_view:
            samples = torchvision.utils.make_grid(X)  # Create image grid
            samples = inverse_normalize(samples, cfg.data.mean, cfg.data.std)  # Revert normalization for display
            logger.add_image('input samples', samples, epoch, dataformats='CHW')  # Log images

        if epoch % cfg.train.eval_interval == 0:  # Run evaluation at interval
            eval(cfg, model, val_loader, cfg.train.criterion, estimator, device)  # Perform validation
            val_scores = estimator.get_scores(6)  # Collect validation metrics
            scores_txt = ['{}: {}'.format(metric, score) for metric, score in val_scores.items()]  # Format metrics
            print_msg('Validation metrics:', scores_txt)  # Print metrics
            if logger:
                for metric, score in val_scores.items():
                    logger.add_scalar('validation {}'.format(metric), score, epoch)  # Log metrics

            indicator = val_scores[cfg.train.indicator]  # Use key metric to track best
            if indicator > max_indicator:
                save_weights(model, os.path.join(cfg.base.save_path, 'best_validation_weights.pt'))  # Save best model
                max_indicator = indicator
                print_msg('Best {} in validation set. Model save at {}'.format(cfg.train.indicator, cfg.base.save_path))  # Notify best model saved

        if epoch % cfg.train.save_interval == 0:
            save_weights(model, os.path.join(cfg.base.save_path, 'epoch_{}.pt'.format(epoch)))  # Save checkpoint

        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if cfg.solver.lr_scheduler == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)  # Scheduler adjusts based on loss
            else:
                lr_scheduler.step()  # Regular scheduler step

    save_weights(model, os.path.join(cfg.base.save_path, 'final_weights.pt'))  # Save final model

    if logger:
        logger.close()  # Close logger session


def evaluate(cfg, model, test_dataset, estimator):
    test_sampler = None  # No special sampling for test set
    test_loader = DataLoader(  # Create DataLoader for test dataset
        test_dataset,
        shuffle=(test_sampler is None),  # Shuffle only if no sampler is used
        sampler=test_sampler,
        batch_size=cfg.train.batch_size,  # Use configured batch size
        num_workers=cfg.train.num_workers,  # Use configured number of data loading workers
        pin_memory=cfg.train.pin_memory  # Use pinned memory to speed up host to GPU transfers
    )

    print('Running on Test set...')
    eval(cfg, model, test_loader, cfg.train.criterion, estimator, cfg.base.device)  # Run evaluation logic

    print('================Finished================')
    test_scores = estimator.get_scores(6)  # Get evaluation metrics
    for metric, score in test_scores.items():  # Print all metrics
        print('{}: {}'.format(metric, score))
    print('Confusion Matrix:')
    print(estimator.get_conf_mat())  # Print confusion matrix
    print('========================================')


def eval(cfg, model, dataloader, criterion, estimator, device):
    model.eval()  # Set model to eval mode (disables dropout, batch norm updates)
    torch.set_grad_enabled(False)  # Turn off autograd for inference (saves memory)

    estimator.reset()  # Clear internal metric states
    for test_data in dataloader:  # Loop over each test batch
        X, y = test_data
        X = X.to(device)  # Move input to device (CPU or GPU)
        y = y.to(device)
        y = select_target_type(y, criterion)  # Format target depending on criterion
        y_pred = model(X)  # Run model inference
        estimator.update(y_pred, y)  # Update metrics based on predictions

    model.train()  # Switch back to train mode
    torch.set_grad_enabled(True)  # Re-enable gradient computation


def initialize_sampler(cfg, train_dataset, val_dataset):
    sampling_strategy = cfg.data.sampling_strategy  # Load sampling strategy from config
    val_sampler = None  # No sampler needed for validation

    if sampling_strategy == 'class_balanced':
        train_sampler = ScheduledWeightedSampler(train_dataset, 1)  # Equal class sampling
    elif sampling_strategy == 'progressively_balanced':
        train_sampler = ScheduledWeightedSampler(train_dataset, cfg.data.sampling_weights_decay_rate)  # Slowly transition to uniform
    elif sampling_strategy == 'instance_balanced':
        train_sampler = None  # Standard random sampling
    else:
        raise NotImplementedError('Not implemented resampling strategy.')  # Invalid config

    return train_sampler, val_sampler


def initialize_dataloader(cfg, train_dataset, val_dataset, train_sampler, val_sampler):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory

    train_loader = DataLoader(  # Dataloader for training
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,  # Drop last batch to keep size consistent
        pin_memory=pin_memory
    )

    val_loader = DataLoader(  # Dataloader for validation
        val_dataset,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=False,  # Keep all batches
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def initialize_loss(cfg, train_dataset):
    criterion = cfg.train.criterion  # Loss name (e.g., cross_entropy)
    criterion_args = cfg.criterion_args[criterion]  # Additional arguments from config

    weight = None  # Optional class weights
    loss_weight_scheduler = None
    loss_weight = cfg.train.loss_weight

    if criterion == 'cross_entropy':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)  # Equal class weights
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, cfg.train.loss_weight_decay_rate)  # Dynamic weights
        elif isinstance(loss_weight, list):  # Manual class weights
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=cfg.base.device)
        loss = nn.CrossEntropyLoss(weight=weight, **criterion_args)
    elif criterion == 'mean_square_error':
        loss = nn.MSELoss(**criterion_args)
    elif criterion == 'mean_absolute_error':
        loss = nn.L1Loss(**criterion_args)
    elif criterion == 'smooth_L1':
        loss = nn.SmoothL1Loss(**criterion_args)
    elif criterion == 'kappa_loss':
        loss = KappaLoss(**criterion_args)
    elif criterion == 'focal_loss':
        loss = FocalLoss(**criterion_args)
    else:
        raise NotImplementedError('Not implemented loss function.')  # Invalid criterion

    loss_function = WarpedLoss(loss, criterion)  # Wrap the loss to apply squeezing or other logic
    return loss_function, loss_weight_scheduler


def initialize_optimizer(cfg, model):
    optimizer_strategy = cfg.solver.optimizer  # e.g., ADAM, SGD
    learning_rate = cfg.solver.learning_rate
    weight_decay = cfg.solver.weight_decay
    momentum = cfg.solver.momentum
    nesterov = cfg.solver.nesterov
    adamw_betas = cfg.solver.adamw_betas

    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAMW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=adamw_betas,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')  # Invalid optimizer choice

    return optimizer


def initialize_lr_scheduler(cfg, optimizer):
    warmup_epochs = cfg.train.warmup_epochs  # Number of warmup epochs
    learning_rate = cfg.solver.learning_rate
    scheduler_strategy = cfg.solver.lr_scheduler

    if not scheduler_strategy:
        lr_scheduler = None  # No scheduler
    else:
        scheduler_args = cfg.scheduler_args[scheduler_strategy]  # Configurable args
        if scheduler_strategy == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'multiple_steps':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
        elif scheduler_strategy == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'clipped_cosine':
            lr_scheduler = ClippedCosineAnnealingLR(optimizer, **scheduler_args)
        else:
            raise NotImplementedError('Not implemented learning rate scheduler.')  # Invalid scheduler

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)  # Add warmup phase
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
