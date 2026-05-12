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
    # LOAD DATA
    # =========================
    train_loader, test_loader, unique_labels = prepare_data()

    # =========================
    # NUMBER OF CLASSES
    # =========================
    num_classes = len(unique_labels)

    # =========================
    # CREATE MODEL
    # =========================
    model = ChordCNN(num_classes)

    # =========================
    # LOSS FUNCTION
    # =========================
    criterion = nn.CrossEntropyLoss()

    # =========================
    # OPTIMIZER
    # =========================
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    EPOCHS = 10

    # =========================
    # MLFLOW TRACKING
    # =========================
    with mlflow.start_run():

        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("num_classes", num_classes)

        # =========================
        # TRAIN MODEL
        # =========================
        model = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            EPOCHS
        )

        # =========================
        # EVALUATE MODEL
        # =========================
        accuracy = evaluate_model(
            model,
            test_loader,
            unique_labels
        )

        # =========================
        # LOG METRICS
        # =========================
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # =========================
        # SAVE MODEL
        # =========================
        mlflow.pytorch.log_model(
            model,
            "model"
        )

        print(f"Final Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()