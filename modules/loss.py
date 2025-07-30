from utils.const import regression_loss  # Import list of regression loss function names

# Define a wrapper class for loss functions
class WarpedLoss():
    def __init__(self, loss_function, criterion):
        self.loss_function = loss_function  # Store the provided loss function
        self.criterion = criterion  # Store the loss criterion name

        self.squeeze = True if self.criterion in regression_loss else False  # Check if the criterion is a regression loss

    def __call__(self, pred, target):  # Define how the class behaves when called like a function
        if self.squeeze:  # If regression loss, remove unnecessary dimensions
            pred = pred.squeeze()  # Squeeze prediction tensor

        return self.loss_function(pred, target)  # Apply the wrapped loss function and return result
