import os
import dagshub
import librosa
import numpy as np
import torch

from dagshub.data_engine import datasources
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================
# AUTHENTICATION (IMPORTANT)
# =========================
dagshub.auth.add_app_token(os.environ["DAGSHUB_TOKEN"])

# =========================
# INIT DAGSHUB
# =========================
dagshub.init(
    repo_owner="thegreatdamsara",
    repo_name="chords",
    mlflow=False
)

# =========================
# LOAD DATASET
# =========================
ds = datasources.get(
    "thegreatdamsara/chords",
    "chord_dataset_real"
)

LOCAL_DATASET_PATH = ds.all().download_files()


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, duration=5)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    target_width = 216

    def adjust_width(data, width):

        if data.shape[1] > width:
            return data[:, :width]

        elif data.shape[1] < width:

            pad_width = width - data.shape[1]

            return np.pad(
                data,
                ((0, 0), (0, pad_width)),
                mode="constant"
            )

        return data

    mel_db = adjust_width(mel_db, target_width)
    chroma = adjust_width(chroma, target_width)

    feature = np.vstack([mel_db, chroma])

    return feature.astype(np.float32)


# =========================
# DATASET CLASS
# =========================
class ChordDataset(Dataset):

    def __init__(self, files, labels):

        self.files = files
        self.labels = labels

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        x = extract_features(self.files[idx])

        x = torch.tensor(x).unsqueeze(0)

        y = torch.tensor(self.labels[idx])

        return x, y


# =========================
# PREPARE DATA
# =========================
def prepare_data(batch_size=32):

    files = []

    for root, dirs, filenames in os.walk(LOCAL_DATASET_PATH):

        for file in filenames:

            if file.endswith(".wav"):

                full_path = os.path.join(root, file)

                files.append(full_path)

    labels = [file.split("/")[-2] for file in files]

    unique_labels = sorted(list(set(labels)))

    label_to_int = {
        label: i
        for i, label in enumerate(unique_labels)
    }

    numeric_labels = [
        label_to_int[label]
        for label in labels
    ]

    train_files, test_files, train_labels, test_labels = train_test_split(
        files,
        numeric_labels,
        test_size=0.2,
        random_state=42,
        stratify=numeric_labels
    )

    train_dataset = ChordDataset(
        train_files,
        train_labels
    )

    test_dataset = ChordDataset(
        test_files,
        test_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        test_loader,
        unique_labels
    )