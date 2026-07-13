import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
#from functools import cache
import kagglehub

try:
    from .models import FFTFeatureDenseModel, ImageFeatureDenseModel
except ImportError:  # Supports running this file directly as a script.
    from models import FFTFeatureDenseModel, ImageFeatureDenseModel


class BaseFeatureDataset(Dataset):
    """Base dataset for loading images and extracting handcrafted descriptors."""

    def __init__(self, dataframe, img_dir, crop_size=32):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.crop_size = crop_size
        self.crop_transform = transforms.CenterCrop(crop_size)
        # DataLoader workers each keep their own process-local feature cache.
        self._feature_cache = {}

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, idx):
        image_id = str(self.dataframe.iloc[idx, 0])
        image_path = os.path.join(self.img_dir, f"{image_id}.tif")
        return Image.open(image_path).convert("RGB")

    def _prepare_crop(self, image):
        """Return a normalized RGB crop without discarding colour information."""
        image = self.crop_transform(image.convert("RGB"))
        return np.asarray(image, dtype=np.float32) / 255.0

    @staticmethod
    def _channels(image_array):
        """Normalize image arrays to an H x W x C representation."""
        if image_array.ndim == 2:
            return image_array[..., np.newaxis]
        if image_array.ndim == 3:
            return image_array
        raise ValueError("Expected a grayscale or RGB image array")

    def _extract_image_features(self, image_array):
        """Extract distribution descriptors independently for each RGB channel."""
        features = []
        percentiles = (1, 5, 10, 25, 50, 75, 90, 95, 99)

        for channel in np.moveaxis(self._channels(image_array), -1, 0):
            mean = float(np.mean(channel))
            centered = channel - mean
            variance = float(np.mean(centered**2))
            if variance <= 1e-12:
                skewness = 0.0
                kurtosis = 0.0
            else:
                skewness = float(np.mean(centered**3) / (variance ** 1.5))
                kurtosis = float(np.mean(centered**4) / (variance**2))

            features.extend(
                [
                    mean,
                    float(np.std(channel)),
                    float(np.min(channel)),
                    float(np.max(channel)),
                    *np.percentile(channel, percentiles).astype(np.float32).tolist(),
                    float(np.mean(channel**2)),
                    skewness,
                    kurtosis,
                ]
            )

        return np.asarray(features, dtype=np.float32)

    def _extract_fft_features(self, image_array):
        """Extract compact, interpretable frequency descriptors per RGB channel."""
        features = []
        for channel in np.moveaxis(self._channels(image_array), -1, 0):
            features.extend(self._extract_channel_fft_features(channel))
        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def _extract_channel_fft_features(channel):
        """Summarize frequency distribution, scale, and orientation for one channel."""
        fft_image = np.fft.fftshift(np.fft.fft2(channel))
        power = np.abs(fft_image) ** 2
        log_power = np.log1p(power)

        h, w = power.shape
        center_y, center_x = h // 2, w // 2
        yy, xx = np.indices((h, w))
        y = yy - center_y
        x = xx - center_x
        radius = np.sqrt(y.astype(np.float32) ** 2 + x.astype(np.float32) ** 2)
        angle = np.arctan2(y, x)

        # Normalize energy-based features so they describe texture rather than brightness.
        total_power = float(np.sum(power))
        normalized_power = power / max(total_power, 1e-12)
        max_radius = float(np.max(radius))

        ring_bins = 5
        wedge_bins = 8
        ring_sizes = np.linspace(0, max_radius + 1e-6, ring_bins + 1)
        ring_features = []
        for i in range(ring_bins):
            low = ring_sizes[i]
            high = ring_sizes[i + 1]
            mask = (radius >= low) & (radius < high)
            ring_features.append(float(np.sum(normalized_power[mask])))

        wedge_features = []
        for i in range(wedge_bins):
            lower = -np.pi + i * (2 * np.pi / wedge_bins)
            upper = -np.pi + (i + 1) * (2 * np.pi / wedge_bins)
            mask = (angle >= lower) & (angle < upper)
            wedge_features.append(float(np.sum(normalized_power[mask])))

        freqy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
        freqx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
        freq_mag = np.sqrt(freqy**2 + freqx**2)
        spectral_centroid = float(np.sum(freq_mag * normalized_power))
        spectral_spread = float(
            np.sqrt(np.sum(((freq_mag - spectral_centroid) ** 2) * normalized_power))
        )
        spectral_entropy = float(-np.sum(normalized_power * np.log(normalized_power + 1e-12)))
        spectral_flatness = float(
            np.exp(np.mean(np.log(power + 1e-12))) / max(float(np.mean(power)), 1e-12)
        )
        high_frequency_ratio = float(np.sum(normalized_power[radius >= 0.5 * max_radius]))
        orientation_coherence = float(np.max(wedge_features) - np.min(wedge_features))

        return np.asarray(
            [
                float(np.mean(log_power)),
                float(np.std(log_power)),
                float(np.median(log_power)),
                float(np.percentile(log_power, 25)),
                float(np.percentile(log_power, 75)),
                float(np.percentile(log_power, 90)),
                float(np.log1p(total_power)),
                spectral_centroid,
                spectral_spread,
                spectral_entropy,
                spectral_flatness,
                high_frequency_ratio,
                orientation_coherence,
                *ring_features,
                *wedge_features,
            ],
            dtype=np.float32,
        )

    def _get_cached_features(self, idx, feature_extractor):
        """Return a cached sample, extracting and storing it on the first access."""
        cached_sample = self._feature_cache.get(idx)
        if cached_sample is not None:
            return cached_sample

        image = self._load_image(idx)
        image_array = self._prepare_crop(image)
        label = int(self.dataframe.iloc[idx, 1])
        features = feature_extractor(image_array)
        sample = (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )
        self._feature_cache[idx] = sample
        return sample


class ImageFeatureDataset(BaseFeatureDataset):
    """Dataset that returns only the statistical image features."""

    def __getitem__(self, idx):
        return self._get_cached_features(idx, self._extract_image_features)


class FFTFeatureDataset(BaseFeatureDataset):
    """Dataset that returns only the FFT-based features."""

    def __getitem__(self, idx):
        return self._get_cached_features(idx, self._extract_fft_features)


def process_data(batch_size=64, mode="full", data_dir=None):
    """
    Download the Kaggle histopathologic dataset, extract handcrafted features from
    32x32 crops, and return separate PyTorch DataLoaders for the image-statistics
    model and the FFT model.
    """
    if data_dir is None:
        if kagglehub is None:
            raise ImportError("kagglehub is required to download the Kaggle dataset. Pass data_dir to use a local dataset instead.")
        data_dir = kagglehub.competition_download("histopathologic-cancer-detection")

    # Use one worker per logical CPU for CPU-bound feature extraction.
    num_workers = os.cpu_count() or 1

    train_dir = os.path.join(data_dir, "train")
    csv_path = os.path.join(data_dir, "train_labels.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find the labels file in {data_dir}")

    df = pd.read_csv(csv_path)

    if mode == "proto":
        print("Mode 'proto' enabled: using 5% of the data for quick experimentation.")
        df, _ = train_test_split(df, train_size=0.05, random_state=42, stratify=df["label"])

    df_train, df_val = train_test_split(df, test_size=0.10, random_state=42, stratify=df["label"])
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print(f"Processed images -> train: {len(df_train)} | validation: {len(df_val)}")

    image_train_dataset = ImageFeatureDataset(dataframe=df_train, img_dir=train_dir)
    image_val_dataset = ImageFeatureDataset(dataframe=df_val, img_dir=train_dir)
    fft_train_dataset = FFTFeatureDataset(dataframe=df_train, img_dir=train_dir)
    fft_val_dataset = FFTFeatureDataset(dataframe=df_val, img_dir=train_dir)

    loader_options = {
        "num_workers": num_workers,
        # Page-locked host memory enables asynchronous CPU-to-CUDA transfers.
        "pin_memory": torch.cuda.is_available(),
        # Keep worker-local __getitem__ caches alive across epochs.
        "persistent_workers": num_workers > 0,
    }
    image_train_loader = DataLoader(image_train_dataset, batch_size=batch_size, shuffle=True, **loader_options)
    image_val_loader = DataLoader(image_val_dataset, batch_size=batch_size, shuffle=False, **loader_options)
    fft_train_loader = DataLoader(fft_train_dataset, batch_size=batch_size, shuffle=True, **loader_options)
    fft_val_loader = DataLoader(fft_val_dataset, batch_size=batch_size, shuffle=False, **loader_options)

    return {
        "image_train": image_train_loader,
        "image_val": image_val_loader,
        "fft_train": fft_train_loader,
        "fft_val": fft_val_loader,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    device=None,
    epochs=5,
    learning_rate=1e-3,
    weight_decay=1e-6,
):
    """Train a simple binary classifier for one feature branch."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [training]",
            # Keep each completed epoch visible instead of clearing it when the
            # next epoch starts.
            leave=True,
        )
        for features, labels in progress_bar:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen_samples += labels.size(0)
            progress_bar.set_postfix(loss=f"{running_loss / seen_samples:.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        metrics = evaluate_model(model, val_loader, device)
        metrics["train_loss"] = train_loss
        history.append(metrics)

    return model, history


def get_predictions(model, data_loader, device=None):
    """Return binary predictions and their aligned targets for a data loader."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device, non_blocking=True)
            outputs = model(features)
            probs = torch.sigmoid(outputs).cpu().squeeze(-1)
            preds = (probs >= 0.5).long().tolist()
            labels = labels.view(-1).long().tolist()
            predictions.extend(preds)
            targets.extend(labels)

    predictions = np.array(predictions)
    targets = np.array(targets)
    return predictions, targets


def evaluate_model(model, data_loader, device=None):
    """Calculate basic classification metrics on a validation loader."""
    predictions, targets = get_predictions(model, data_loader, device)

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }


def calculate_yules_q(first_predictions, second_predictions, targets):
    """Measure association between two models' validation-set mistakes.

    Q = 0 indicates uncorrelated errors, Q > 0 indicates that the models tend to
    make mistakes on the same examples, and Q < 0 indicates complementary errors.
    """
    first_predictions = np.asarray(first_predictions)
    second_predictions = np.asarray(second_predictions)
    targets = np.asarray(targets)

    if not (
        first_predictions.shape == second_predictions.shape == targets.shape
    ):
        raise ValueError("Predictions and targets must have the same shape")

    first_errors = first_predictions != targets
    second_errors = second_predictions != targets
    both_wrong = int(np.sum(first_errors & second_errors))
    both_correct = int(np.sum(~first_errors & ~second_errors))
    first_only_wrong = int(np.sum(first_errors & ~second_errors))
    second_only_wrong = int(np.sum(~first_errors & second_errors))

    numerator = both_wrong * both_correct - first_only_wrong * second_only_wrong
    denominator = both_wrong * both_correct + first_only_wrong * second_only_wrong
    yules_q = float(numerator / denominator) if denominator else float("nan")

    return {
        "yules_q": yules_q,
        "both_wrong": both_wrong,
        "both_correct": both_correct,
        "first_only_wrong": first_only_wrong,
        "second_only_wrong": second_only_wrong,
    }


def train_and_compare_models(
    batch_size=64,
    mode="full",
    data_dir=None,
    epochs=5,
    learning_rate=1e-3,
    weight_decay=1e-6,
    device=None,
):
    """Train both dense feature models and compare their validation metrics."""
    loaders = process_data(batch_size=batch_size, mode=mode, data_dir=data_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    image_model = ImageFeatureDenseModel().to(device)
    print(next(image_model.parameters()).device)
    fft_model = FFTFeatureDenseModel().to(device)

    image_model, image_history = train_model(
        image_model,
        loaders["image_train"],
        loaders["image_val"],
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    fft_model, fft_history = train_model(
        fft_model,
        loaders["fft_train"],
        loaders["fft_val"],
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    image_metrics = image_history[-1] if image_history else {}
    fft_metrics = fft_history[-1] if fft_history else {}
    image_predictions, image_targets = get_predictions(
        image_model, loaders["image_val"], device
    )
    fft_predictions, fft_targets = get_predictions(fft_model, loaders["fft_val"], device)
    if not np.array_equal(image_targets, fft_targets):
        raise RuntimeError("Validation targets are not aligned between the two models")
    error_association = calculate_yules_q(
        image_predictions, fft_predictions, image_targets
    )

    print("\nModel comparison")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Loss':<10}")
    print(f"{'Image statistics':<20} {image_metrics.get('accuracy', 0):<10.4f} {image_metrics.get('precision', 0):<10.4f} {image_metrics.get('recall', 0):<10.4f} {image_metrics.get('f1', 0):<10.4f} {image_metrics.get('train_loss', 0):<10.4f}")
    print(f"{'FFT features':<20} {fft_metrics.get('accuracy', 0):<10.4f} {fft_metrics.get('precision', 0):<10.4f} {fft_metrics.get('recall', 0):<10.4f} {fft_metrics.get('f1', 0):<10.4f} {fft_metrics.get('train_loss', 0):<10.4f}")
    print(
        "Yule's Q (model errors): "
        f"{error_association['yules_q']:.4f} "
        f"(both wrong: {error_association['both_wrong']}, "
        f"image only: {error_association['first_only_wrong']}, "
        f"FFT only: {error_association['second_only_wrong']})"
    )

    return {
        "image_model": image_model,
        "fft_model": fft_model,
        "image_history": image_history,
        "fft_history": fft_history,
        "image_metrics": image_metrics,
        "fft_metrics": fft_metrics,
        "error_association": error_association,
        "image_predictions": image_predictions,
        "fft_predictions": fft_predictions,
        "validation_targets": image_targets,
        "validation_ids": loaders["image_val"].dataset.dataframe.iloc[:, 0]
        .astype(str)
        .to_numpy(),
    }

if __name__ == "__main__":
    results = train_and_compare_models(batch_size=64, mode="full", epochs=3, learning_rate=1e-2)
