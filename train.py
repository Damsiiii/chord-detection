import os
import mlflow

# =========================
# MLflow Setup (IMPORTANT)
# =========================

mlflow.set_tracking_uri(
    "https://dagshub.com/thegreatdamsara/chords.mlflow/#/"
)

# GitHub Secrets (works in GitHub Actions)
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")


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

    # 🔥 THIS IS REQUIRED OR NOTHING WILL SHOW IN MLflow
    with mlflow.start_run():

        # log hyperparameters once
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("model_type", str(type(model).__name__))

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

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

            # log metric
            mlflow.log_metric(
                "train_loss",
                epoch_loss,
                step=epoch
            )

        # OPTIONAL: save model
        model_path = "model.pth"
        try:
            import torch
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
        except Exception as e:
            print("Model save skipped:", e)

    return model