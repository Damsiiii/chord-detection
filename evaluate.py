import torch
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================
# EVALUATE FUNCTION
# =========================
def evaluate_model(
    model,
    test_loader,
    unique_labels
):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for X, y in test_loader:

            outputs = model(X)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=unique_labels,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    return accuracy