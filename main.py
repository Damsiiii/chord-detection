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

    train_loader, test_loader = prepare_data()

    model = ChordCNN()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    EPOCHS = 10

    with mlflow.start_run():

        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", 0.001)

        model = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            EPOCHS
        )

        accuracy = evaluate_model(
            model,
            test_loader
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.pytorch.log_model(
            model,
            "model"
        )


if __name__ == "__main__":
    main()