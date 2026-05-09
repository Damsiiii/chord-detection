{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP9oDVXN2IxgBUSyKtJjz3t",
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
        "<a href=\"https://colab.research.google.com/github/Damsiiii/chord-detection/blob/main/train.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Kn8SpniTjH3s"
      },
      "outputs": [],
      "source": [
        "# TRAIN LOOP\n",
        "# =========================\n",
        "import torch\n",
        "from torch.utils.data import DataLoader\n",
        "import mlflow\n",
        "\n",
        "# Re-initialize dataset and loader to ensure they use the updated extract_features with padding/truncation\n",
        "# This ensures all tensors in the batch are [1, 140, 216]\n",
        "train_dataset = ChordDataset(files, numeric_labels[:len(files)])\n",
        "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
        "\n",
        "for epoch in range(EPOCHS):\n",
        "    model.train()\n",
        "    running_loss = 0.0\n",
        "    for X, y in train_loader:\n",
        "        outputs = model(X)\n",
        "        loss = criterion(outputs, y)\n",
        "        optimizer.zero_grad()\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        running_loss += loss.item()\n",
        "\n",
        "    epoch_loss = running_loss / len(train_loader)\n",
        "    print(f\"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}\")\n",
        "    mlflow.log_metric(\"loss\", epoch_loss, step=epoch)"
      ]
    }
  ]
}