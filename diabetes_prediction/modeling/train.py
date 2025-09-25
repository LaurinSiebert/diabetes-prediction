import torch
import numpy as np
import random
from diabetes_prediction import config
from diabetes_prediction.dataset import get_dataloader
from diabetes_prediction.modeling.model import DiabetesModel

def set_seed(seed : int = config.SEED):
    """Sets the random seed for the training. Likely more than necessary."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# def criterion

# def evaluation


def train():
    """Training Loop"""

    # Set the seed for reproducability
    set_seed()

    # Initialize the dataloader. path default is the raw data, batch_size default is specified in config.py
    train_dl, test_dl, val_dl = get_dataloader()

    # Initialize the model
    input_size = iter(next(train_dl)).size[0] #<- clean up
    model = DiabetesModel(input_size, config.HIDDEN_SIZE)
    if torch.cuda.is_available:
        model.to_device(...) #<- device!

    # Define an optimizer and a loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("Starting training ... \n")

    # Training loop
    for epoch in range(config.EPOCHS):
        model.train()

        epoch_loss = 0.0
        #epoch_acc = 0.0

        for X, y in train_dl:
            X, y = X.to_device(), y.to_device() #<- clean up

            optimizer.zero_grad()
            outputs = model(X).squeeze()

            loss = criterion(outputs, y)
            #acc = binary_accuracy(outputs, y)

            loss.backwards()
            optimizer.step()

            epoch_loss += loss.item()
            #epoch_acc += acc.item()

        avg_loss = epoch_loss / len(train_dl)
        #avg_acc = epoch_acc / len(train_dl)

        #val_loss, val_acc = evaluate(model, val_loader, criterion) <- needs to be done

        print(
            f"Epoch {epoch}/{config.EPOCHS}"
            #f"Train loss: {avg_loss:.4f} | Train acc: {avg_acc:.4f}"
            #f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        #Final evaluation

        # Safe the model.


if __name__ == "__main__":
    train()
