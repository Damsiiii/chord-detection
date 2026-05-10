# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from preprocessing import prepare_data
from model import ChordCNN
from train import train_model
from evaluate import evaluate_model


def main():

    # =========================
    # PREPARE DATA
    # =========================
    train_loader, test_loader = prepare_data()

    # =========================
    # CREATE MODEL
    # =========================
    model = ChordCNN()

    # =========================
    # LOSS + OPTIMIZER
    # =========================
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    EPOCHS = 10

#ML flow tracking
        with mlflow.start_run():

        # Log hyperparameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("optimizer", "Adam")

#Train model
        model = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            EPOCHS
        )

#Evaluate
        accuracy = evaluate_model(
            model,
            test_loader
        )

        print(f"Accuracy: {accuracy:.4f}")

        mlflow.log_metric("accuracy", accuracy)

#Save the model
        mlflow.pytorch.log_model(
            model,
            "model"
        )

#Entry point
if __name__ == "__main__":
    main()