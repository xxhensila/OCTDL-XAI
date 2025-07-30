import torch  # PyTorch for tensor operations
import torcheval.metrics as tm  # PyTorch evaluation metrics
from utils.func import print_msg  # Custom print function with formatting

# Estimator class to compute and manage evaluation metrics
class Estimator():
    def __init__(self, metrics, num_classes, criterion, average='macro', thresholds=None):
        self.criterion = criterion  # Store criterion name (e.g., 'cross_entropy')
        self.num_classes = num_classes  # Number of output classes
        self.thresholds = [-0.5 + i for i in range(num_classes)] if not thresholds else thresholds  # Thresholds for regression-based classification

        # Warn and remove AUC if not applicable to regression
        if criterion in regression_based_metrics and 'auc' in metrics:
            metrics.remove('auc')  # Remove AUC from metrics
            print_msg('AUC is not supported for regression based metrics {}.'.format(criterion), warning=True)  # Warning message

        self.metrics = metrics  # Save metric names
        self.metrics_fn = {m: metrics_fn[m](num_classes=num_classes, average=average) for m in metrics}  # Initialize metric functions
        self.conf_mat_fn = tm.MulticlassConfusionMatrix(num_classes=num_classes)  # Confusion matrix tracker

    # Update metrics with new batch of predictions and targets
    def update(self, predictions, targets):
        targets = targets.data.cpu().long()  # Move targets to CPU and cast to long
        logits = predictions.data.cpu()  # Move predictions (logits) to CPU
        predictions = self.to_prediction(logits)  # Convert logits to final class predictions

        self.conf_mat_fn.update(predictions, targets)  # Update confusion matrix
        for m in self.metrics_fn.keys():  # For each metric
            if m in logits_required_metrics:  # Use raw logits if needed (e.g., AUC)
                self.metrics_fn[m].update(logits, targets)
            else:  # Use class predictions otherwise
                self.metrics_fn[m].update(predictions, targets)

    # Get dictionary of all computed scores
    def get_scores(self, digits=-1):
        scores = {m: self._compute(m, digits) for m in self.metrics}  # Compute each metric
        return scores  # Return dictionary of scores

    # Compute a single metric
    def _compute(self, metric, digits=-1):
        score = self.metrics_fn[metric].compute().item()  # Compute and extract value
        score = score if digits == -1 else round(score, digits)  # Round if needed
        return score  # Return metric score

    # Get confusion matrix as numpy array
    def get_conf_mat(self):
        return self.conf_mat_fn.compute().numpy().astype(int)  # Return confusion matrix

    # Reset all metrics
    def reset(self):
        for m in self.metrics_fn.keys():  # Reset each metric
            self.metrics_fn[m].reset()
        self.conf_mat_fn.reset()  # Reset confusion matrix

    # Convert logits or regression output to class predictions
    def to_prediction(self, predictions):
        if self.criterion in regression_based_metrics:  # If regression-based
            predictions = torch.tensor([self.classify(p.item()) for p in predictions]).long()  # Classify each scalar
        else:
            predictions = torch.argmax(predictions, dim=1).long()  # Use argmax for classification
        return predictions  # Return predictions

    # Convert regression output to discrete class using thresholds
    def classify(self, predict):
        thresholds = self.thresholds  # Get thresholds
        predict = max(predict, thresholds[0])  # Clamp to min threshold
        for i in reversed(range(len(thresholds))):  # Reverse search for highest matching class
            if predict >= thresholds[i]:
                return i  # Return corresponding class

# Dictionary mapping metric names to their implementation
metrics_fn = {
    'acc': tm.MulticlassAccuracy,  # Accuracy
    'f1': tm.MulticlassF1Score,  # F1-score
    'auc': tm.MulticlassAUROC,  # Area under ROC curve
    'precision': tm.MulticlassPrecision,  # Precision
    'recall': tm.MulticlassRecall  # Recall
}

available_metrics = metrics_fn.keys()  # All supported metrics
logits_required_metrics = ['auc']  # Metrics that require logits instead of class predictions
regression_based_metrics = ['mean_square_error', 'mean_absolute_error', 'smooth_L1']  # Losses for regression
