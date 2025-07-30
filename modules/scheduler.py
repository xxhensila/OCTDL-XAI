import torch  # PyTorch for tensor operations and optimization
from torch.utils.data.sampler import Sampler  # Base class for custom sampling

# Warmup learning rate scheduler
class WarmupLRScheduler():
    def __init__(self, optimizer, warmup_epochs, initial_lr):
        self.epoch = 0  # Initialize epoch counter
        self.optimizer = optimizer  # Store optimizer
        self.warmup_epochs = warmup_epochs  # Number of warmup epochs
        self.initial_lr = initial_lr  # Target learning rate after warmup

    def step(self):  # Update learning rate each epoch
        if self.epoch <= self.warmup_epochs:  # Only update during warmup
            self.epoch += 1  # Increment epoch
            curr_lr = (self.epoch / self.warmup_epochs) * self.initial_lr  # Linearly scale LR
            for param_group in self.optimizer.param_groups:  # Update each param group
                param_group['lr'] = curr_lr  # Set new learning rate

    def is_finish(self):  # Check if warmup is finished
        return self.epoch >= self.warmup_epochs  # Return True if warmup is done


# Custom weighted sampler with dynamic decay
class ScheduledWeightedSampler(Sampler):
    def __init__(self, dataset, decay_rate):
        self.dataset = dataset  # Store dataset
        self.decay_rate = decay_rate  # Decay rate for class weights

        self.num_samples = len(dataset)  # Total number of samples
        self.targets = [sample[1] for sample in dataset.imgs]  # Extract target labels
        self.class_weights = self.cal_class_weights()  # Compute initial class weights

        self.epoch = 0  # Epoch counter
        self.w0 = torch.as_tensor(self.class_weights, dtype=torch.double)  # Initial weights
        self.wf = torch.as_tensor([1] * len(self.dataset.classes), dtype=torch.double)  # Final uniform weights
        self.sample_weight = torch.zeros(self.num_samples, dtype=torch.double)  # Sample weights init
        for i, _class in enumerate(self.targets):  # Assign weights per sample
            self.sample_weight[i] = self.w0[_class]  # Use class weight

    def step(self):  # Update weights based on epoch and decay
        if self.decay_rate < 1:  # Only if decay applies
            self.epoch += 1  # Increment epoch
            factor = self.decay_rate**(self.epoch - 1)  # Calculate decay factor
            self.weights = factor * self.w0 + (1 - factor) * self.wf  # Interpolate between w0 and wf
            for i, _class in enumerate(self.targets):  # Update sample weights
                self.sample_weight[i] = self.weights[_class]  # Set updated weight

    def __iter__(self):  # Sampling iterator
        return iter(torch.multinomial(self.sample_weight, self.num_samples, replacement=True).tolist())  # Sample with replacement

    def __len__(self):  # Length of sampler
        return self.num_samples  # Total number of samples

    def cal_class_weights(self):  # Compute initial class weights
        num_classes = len(self.dataset.classes)  # Number of classes
        classes_idx = list(range(num_classes))  # Class indices
        class_count = [self.targets.count(i) for i in classes_idx]  # Count samples per class
        weights = [self.num_samples / class_count[i] for i in classes_idx]  # Inverse frequency weighting
        min_weight = min(weights)  # Normalize to smallest weight
        class_weights = [weights[i] / min_weight for i in classes_idx]  # Normalize all weights
        return class_weights  # Return normalized class weights


# Scheduler for dynamically adjusting loss weights
class LossWeightsScheduler():
    def __init__(self, dataset, decay_rate):
        self.dataset = dataset  # Store dataset
        self.decay_rate = decay_rate  # Store decay rate

        self.num_samples = len(dataset)  # Total number of samples
        self.targets = [sample[1] for sample in dataset.imgs]  # Extract target labels
        self.class_weights = self.cal_class_weights()  # Compute initial class weights

        self.epoch = 0  # Epoch counter
        self.w0 = torch.as_tensor(self.class_weights, dtype=torch.float32)  # Initial weights
        self.wf = torch.as_tensor([1] * len(self.dataset.classes), dtype=torch.float32)  # Final uniform weights

    def step(self):  # Update and return current class weights
        weights = self.w0  # Default to initial weights
        if self.decay_rate < 1:  # If using decay
            self.epoch += 1  # Increment epoch
            factor = self.decay_rate**(self.epoch - 1)  # Compute decay factor
            weights = factor * self.w0 + (1 - factor) * self.wf  # Interpolate between w0 and wf
        return weights  # Return updated weights

    def __len__(self):  # Length method
        return self.num_samples  # Return number of samples

    def cal_class_weights(self):  # Compute weights per class
        num_classes = len(self.dataset.classes)  # Number of classes
        classes_idx = list(range(num_classes))  # Class indices
        class_count = [self.targets.count(i) for i in classes_idx]  # Sample count per class
        weights = [self.num_samples / class_count[i] for i in classes_idx]  # Inverse frequency
        min_weight = min(weights)  # Find minimum weight
        class_weights = [weights[i] / min_weight for i in classes_idx]  # Normalize weights
        return class_weights  # Return weights


# Cosine annealing LR scheduler with minimum clipping
class ClippedCosineAnnealingLR():
    def __init__(self, optimizer, T_max, min_lr):
        self.optimizer = optimizer  # Store optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)  # Built-in scheduler
        self.min_lr = min_lr  # Minimum learning rate
        self.finish = False  # Whether LR reached minimum

    def step(self):  # Perform one scheduler step
        if not self.finish:  # If not yet finished
            self.scheduler.step()  # Step built-in scheduler
            curr_lr = self.optimizer.param_groups[0]['lr']  # Get current learning rate
            if curr_lr < self.min_lr:  # If below threshold
                for param_group in self.optimizer.param_groups:  # Set all LRs
                    param_group['lr'] = self.min_lr  # Clip to minimum
                self.finish = True  # Mark as finished

    def is_finish(self):  # Check if scheduler is finished
        return self.finish  # Return finish status
