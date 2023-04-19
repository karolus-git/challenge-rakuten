import torchmetrics

def acc(num_labels):
    """
    The acc function returns a function that computes the accuracy of a model.
    
    The returned function takes in two arguments: predictions and targets, both of which are tensors. 
    It returns the accuracy as a float between 0 and 1.
    
    :param num_labels: Used to Specify the number of classes in the dataset.
    :return: A function that takes in the predictions and labels.
    
    :doc-author: Trelent
    """
    return torchmetrics.Accuracy(
            task="multiclass", 
            average="weighted",
            num_classes=num_labels)

def val_acc(num_labels):
    """
    The val_acc function returns a function that calculates the accuracy of a model's predictions.
    
    The returned function takes in two arguments: 
    
        1) The model's predictions (a tensor of shape [batch_size, num_classes]) and 
    
        2) The true labels (a tensor of shape [batch_size]).
        
    It returns the accuracy as a float between 0 and 1.0.
    
    :param num_labels: Used to Specify the number of classes in the dataset.
    :return: The accuracy of the model.
    
    :doc-author: Trelent
    """
    return torchmetrics.Accuracy(
            task="multiclass", 
            average="weighted",
            num_classes=num_labels)

def loss(num_labels):
    """
    The loss function for the hinge loss.
    
    The hinge loss is defined as:
    
        .. math :: L_i = \sum_{j \neq y_i} [max(0, f_j - f_{y_i} + 1)]^p
    
    :param num_labels: Used to Specify the number of classes in the dataset.
    :return: A scalar.
    
    :doc-author: Trelent
    """
    return torchmetrics.HingeLoss(
            task="multiclass", 
            num_classes=num_labels)

def val_loss(num_labels):
    """
    The val_loss function is used to calculate the loss of a model on validation data.
    
    The val_loss function is called by the trainer after each epoch, and it takes in 
    the model's predictions for all validation examples as well as their true labels. 
    It returns a single value representing the total loss across all examples in the 
    validation set.
    
    :param num_labels: Used to Specify the number of classes in the dataset.
    :return: A function that takes in the predictions and labels.
    
    :doc-author: Trelent
    """
    return torchmetrics.HingeLoss(
            task="multiclass", 
            num_classes=num_labels)