{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO9GGLp0J8dxM87Bkz9hmtb",
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
        "<a href=\"https://colab.research.google.com/github/Damsiiii/chord-detection/blob/main/preprocessing.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Dataset loading from dagshub"
      ],
      "metadata": {
        "id": "OEzx0csBgfZT"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8LV820DogYSv"
      },
      "outputs": [],
      "source": [
        "dagshub.init(repo_owner='thegreatdamsara', repo_name='chords', mlflow=False)\n",
        "\n",
        "from dagshub.data_engine import datasources\n",
        "ds = datasources.get('thegreatdamsara/chords', 'chord_dataset_real')"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Feature extraction"
      ],
      "metadata": {
        "id": "Em9vbX9Xglja"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import librosa\n",
        "import numpy as np\n",
        "\n",
        "def extract_features(file_path):\n",
        "    # Load audio\n",
        "    y, sr = librosa.load(file_path, duration=5)\n",
        "\n",
        "    # Generate Mel Spectrogram\n",
        "    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)\n",
        "    mel_db = librosa.power_to_db(mel)\n",
        "\n",
        "    # Generate Chroma\n",
        "    chroma = librosa.feature.chroma_stft(y=y, sr=sr)\n",
        "\n",
        "    # Target width for consistency\n",
        "    target_width = 216\n",
        "\n",
        "    def adjust_width(data, width):\n",
        "        if data.shape[1] > width:\n",
        "            # Truncate\n",
        "            return data[:, :width]\n",
        "        elif data.shape[1] < width:\n",
        "            # Pad with zeros\n",
        "            pad_width = width - data.shape[1]\n",
        "            return np.pad(data, ((0, 0), (0, pad_width)), mode='constant')\n",
        "        return data\n",
        "\n",
        "    mel_db = adjust_width(mel_db, target_width)\n",
        "    chroma = adjust_width(chroma, target_width)\n",
        "\n",
        "    # Stack vertically: 128 (mel) + 12 (chroma) = 140 rows\n",
        "    feature = np.vstack([mel_db, chroma])\n",
        "\n",
        "    return feature.astype(np.float32)"
      ],
      "metadata": {
        "id": "OAhpnkIfgmMO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Dataset class"
      ],
      "metadata": {
        "id": "IokniugNgpjF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from torch.utils.data import Dataset\n",
        "import torch\n",
        "import os\n",
        "\n",
        "\n",
        "class ChordDataset(Dataset):\n",
        "\n",
        "    def __init__(self, files, labels):\n",
        "        self.files = files\n",
        "        self.labels = labels\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.files)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "\n",
        "        x = extract_features(self.files[idx])\n",
        "\n",
        "        x = torch.tensor(x).unsqueeze(0)\n",
        "\n",
        "        y = torch.tensor(self.labels[idx])\n",
        "\n",
        "        return x, y"
      ],
      "metadata": {
        "id": "YUvBGY1SgsAt"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}