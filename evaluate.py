{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMCalU3kOovnV1j2qmtfzd7",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Damsiiii/chord-detection/blob/main/evaluate.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "BSRiESfmJTDP"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import (\n",
        "    accuracy_score,\n",
        "    precision_score,\n",
        "    recall_score,\n",
        "    f1_score,\n",
        "    confusion_matrix,\n",
        "    classification_report\n",
        ")\n",
        "\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import mlflow\n",
        "\n",
        "# =========================\n",
        "# EVALUATION\n",
        "# =========================\n",
        "model.eval()\n",
        "\n",
        "all_preds = []\n",
        "all_labels = []\n",
        "\n",
        "with torch.no_grad():\n",
        "\n",
        "    for X, y in train_loader:\n",
        "\n",
        "        outputs = model(X)\n",
        "\n",
        "        preds = torch.argmax(outputs, dim=1)\n",
        "\n",
        "        all_preds.extend(preds.cpu().numpy())\n",
        "        all_labels.extend(y.cpu().numpy())\n",
        "\n",
        "# =========================\n",
        "# METRICS\n",
        "# =========================\n",
        "accuracy = accuracy_score(all_labels, all_preds)\n",
        "\n",
        "precision = precision_score(\n",
        "    all_labels,\n",
        "    all_preds,\n",
        "    average='weighted',\n",
        "    zero_division=0\n",
        ")\n",
        "\n",
        "recall = recall_score(\n",
        "    all_labels,\n",
        "    all_preds,\n",
        "    average='weighted',\n",
        "    zero_division=0\n",
        ")\n",
        "\n",
        "f1 = f1_score(\n",
        "    all_labels,\n",
        "    all_preds,\n",
        "    average='weighted',\n",
        "    zero_division=0\n",
        ")\n",
        "\n",
        "print(f\"Accuracy : {accuracy:.4f}\")\n",
        "print(f\"Precision: {precision:.4f}\")\n",
        "print(f\"Recall   : {recall:.4f}\")\n",
        "print(f\"F1 Score : {f1:.4f}\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# =========================\n",
        "# CLASSIFICATION REPORT\n",
        "# =========================\n",
        "print(\"\\nClassification Report:\\n\")\n",
        "\n",
        "print(\n",
        "    classification_report(\n",
        "        all_labels,\n",
        "        all_preds,\n",
        "        target_names=unique_labels,\n",
        "        zero_division=0\n",
        "    )\n",
        ")"
      ],
      "metadata": {
        "id": "t4N9i8KHNAfc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# =========================\n",
        "# CONFUSION MATRIX\n",
        "# =========================\n",
        "cm = confusion_matrix(all_labels, all_preds)\n",
        "\n",
        "plt.figure(figsize=(16, 14))\n",
        "\n",
        "sns.heatmap(\n",
        "    cm,\n",
        "    annot=False,\n",
        "    cmap=\"Blues\",\n",
        "    xticklabels=unique_labels,\n",
        "    yticklabels=unique_labels\n",
        ")\n",
        "\n",
        "plt.xlabel(\"Predicted\")\n",
        "plt.ylabel(\"True\")\n",
        "plt.title(\"Chord Confusion Matrix\")\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "P6bK6SizNDHt"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}