{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNMs73loXH/FTRi6WTQKvhM",
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
    },
    {
      "cell_type": "markdown",
      "source": [
        "Prepare dataset and dataloader"
      ],
      "metadata": {
        "id": "Lv3QjvONj2pZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "    # Prepare dataset and dataloader\n",
        "    files = [entry.path for entry in ds.all()]\n",
        "    labels = [file.split('/')[-2] for file in files] # Assuming label is parent folder name\n",
        "\n",
        "    # Get unique labels and map them to integers from 0 to num_classes-1\n",
        "    unique_labels = sorted(list(set(labels)))\n",
        "    label_to_int = {label: i for i, label in enumerate(unique_labels)}\n",
        "    numeric_labels = [label_to_int[label] for label in labels]"
      ],
      "metadata": {
        "id": "f1a9sq93kCqV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Load the dataset"
      ],
      "metadata": {
        "id": "r9PqNhOjk2py"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import dagshub\n",
        "from dagshub.data_engine import datasources\n",
        "\n",
        "# =========================\n",
        "# INIT DAGSHUB\n",
        "# =========================\n",
        "dagshub.init(\n",
        "    repo_owner='thegreatdamsara',\n",
        "    repo_name='chords',\n",
        "    mlflow=False\n",
        ")\n",
        "\n",
        "# =========================\n",
        "# LOAD DATASOURCE\n",
        "# =========================\n",
        "ds = datasources.get(\n",
        "    'thegreatdamsara/chords',\n",
        "    'chord_dataset_real'\n",
        ")"
      ],
      "metadata": {
        "id": "AA21T0euk4w7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# =========================\n",
        "# DOWNLOAD DATASET\n",
        "# =========================\n",
        "# We use ds.all() to get a QueryResult, which has the download_files method.\n",
        "LOCAL_DATASET_PATH = ds.all().download_files()\n",
        "\n",
        "print(\"✅ Dataset downloaded to:\")\n",
        "print(LOCAL_DATASET_PATH)"
      ],
      "metadata": {
        "id": "xfAgIPDnlAci"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# BUILD FILE LIST\n",
        "# =========================\n",
        "files = []\n",
        "\n",
        "for root, dirs, filenames in os.walk(LOCAL_DATASET_PATH):\n",
        "\n",
        "    for file in filenames:\n",
        "\n",
        "        if file.endswith(\".wav\"):\n",
        "\n",
        "            full_path = os.path.join(root, file)\n",
        "\n",
        "            files.append(full_path)\n",
        "\n",
        "print(f\"✅ Total WAV files found: {len(files)}\")\n",
        "\n",
        "# =========================\n",
        "# VERIFY\n",
        "# =========================\n",
        "print(files[:5])"
      ],
      "metadata": {
        "id": "cXm3R1uQlHha"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}