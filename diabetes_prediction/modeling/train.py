import torch
import numpy as np
import random
import os
from diabetes_prediction import config
from diabetes_prediction.dataset import get_dataloader
from diabetes_prediction.modeling.model import DiabetesModel

def set_seed(seed : int = config.SEED):
    """
    Sets the random seed for the training. Likely more than necessary.

    Args:
          seed : Random Seed, specified in config.py
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def binary_accuracy(preds, targets, threshold=0.5):
    """
    Computes binary classification accuracy.

    Args:
        preds : Predicted probabilities (between 0 and 1)
        targets : Ground truth labels (0 or 1)
        threshold : Threshold to convert probabilities to binary predictions

    Returns:
        float: Accuracy (between 0 and 1)
    """

    # Apply Sigmoid to convert Logits to values [0,1]
    preds = torch.sigmoid(preds)

    # Convert predicted probabilities to binary labels (0 or 1)
    pred_labels = (preds >= threshold).int()

    # Compare with ground truth
    correct = (pred_labels == targets).sum()

    # Compute accuracy
    accuracy = correct / len(targets)

    return accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()  # set model to eval mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # disable gradient calculation
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze(1)

            # Apply sigmoid
            probs = torch.sigmoid(outputs)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # sum loss over batch

            # Compute binary accuracy
            preds = (probs >= 0.5).int()
            correct = (preds == labels).int().sum()
            total_correct += correct.item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy



def train():
    """Training Loop"""

    # Set the seed for reproducability
    set_seed()

    device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")

    # Initialize the dataloader. path default is the raw data, batch_size default is specified in config.py
    train_dl, test_dl, val_dl = get_dataloader()

    # Initialize the model
    input_size = next(iter(train_dl))[0].shape[1]
    model = DiabetesModel(input_size, config.HIDDEN_SIZE)
    if torch.cuda.is_available():
        model.to(device)

    # Define an optimizer and a loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Initialize the best val_loss to be infinite
    best_val_loss = float("inf")

    print("Starting training ... \n")

    # Training loop
    for epoch in range(config.EPOCHS):
        model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0

        for X, y in train_dl:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X).squeeze(1)

            loss = criterion(outputs, y)
            acc = binary_accuracy(outputs, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        avg_loss = epoch_loss / len(train_dl)
        avg_acc = epoch_acc / len(train_dl)

        val_loss, val_acc = evaluate(model, val_dl, criterion, device)

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS}\n"
            f"Train loss: {avg_loss:.4f} | Train acc: {avg_acc:.4f}\n"
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}\n"
        )

        # Saving the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_during_training.pth')

    # Load the best model state
    best_model = DiabetesModel(input_size, config.HIDDEN_SIZE)
    best_model.load_state_dict(torch.load("best_model_during_training.pth"))
    best_model.to(device)

    # Final evaluation on the test set in comparison to train and val set
    train_loss, train_acc = evaluate(best_model, train_dl, criterion, device)
    val_loss, val_acc = evaluate(best_model, val_dl, criterion, device)
    test_loss, test_acc = evaluate(best_model, test_dl, criterion, device)

    print(
        f"Final models loss and acc in comparison:\n"
        f"---------------loss | acc\n"
        f"Train set: {train_loss:.4f} | {train_acc:.4f}\n"
        f"Val set: {val_loss:.4f} | {val_acc:.4f}\n"
        f"Test set: {test_loss:.4f} | {test_acc:.4f}\n"
    )

    # Save the final model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    save_path = os.path.join(config.MODELS_DIR, "final_model.pth")
    torch.save(best_model.state_dict(), save_path)


if __name__ == "__main__":
    train()
