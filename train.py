import mlflow


# =========================
# TRAIN FUNCTION
# =========================
def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs
):

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for X, y in train_loader:

            outputs = model(X)

            loss = criterion(outputs, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {epoch_loss:.4f}"
        )

        mlflow.log_metric(
            "loss",
            epoch_loss,
            step=epoch
        )

    return model