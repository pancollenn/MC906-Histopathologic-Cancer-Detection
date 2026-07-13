"""Train the handcrafted-feature models and CNN on one shared validation split."""

import kagglehub
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cnn_resnet import data_setup as cnn_data_setup
from cnn_resnet import engine as cnn_engine
from cnn_resnet import evaluate as cnn_evaluate
from cnn_resnet.model_builder import BaselineCNN, get_resnet18_model
from ensemble.train import calculate_yules_q, train_and_compare_models


def _print_yules_q(label, association):
    print(
        f"Yule's Q ({label} errors): {association['yules_q']:.4f} "
        f"(both wrong: {association['both_wrong']}, "
        f"first only: {association['first_only_wrong']}, "
        f"second only: {association['second_only_wrong']})"
    )


def train_and_compare_all_models(
    batch_size=64,
    mode="full",
    data_dir=None,
    epochs=5,
    learning_rate=1e-3,
    weight_decay=1e-6,
    cnn_model_type="resnet",
    cnn_epochs=None,
    cnn_learning_rate=1e-3,
    cnn_weight_decay=1e-6,
    cnn_num_workers=2,
    plot_examples=True,
    device=None,
):
    """Train image-statistics, FFT, and CNN models and compare their errors.

    The data loaders in both folders use the same stratified split (seed 42).
    Validation image IDs are checked before calculating Yule's Q, ensuring every
    Q value compares predictions for the exact same images.
    """
    if data_dir is None:
        data_dir = kagglehub.competition_download("histopathologic-cancer-detection")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cnn_epochs is None:
        cnn_epochs = epochs

    cnn_data_setup.set_seed(42)
    feature_results = train_and_compare_models(
        batch_size=batch_size,
        mode=mode,
        data_dir=data_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        plot_examples=plot_examples,
        device=device,
    )

    cnn_data_setup.set_seed(42)
    cnn_train_loader, cnn_val_loader = cnn_data_setup.create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=cnn_num_workers,
        mode=mode,
    )
    if cnn_model_type == "baseline":
        cnn_model = BaselineCNN()
    elif cnn_model_type == "resnet":
        cnn_model = get_resnet18_model(pretrained=True, fine_tune=True)
    else:
        raise ValueError("cnn_model_type must be 'baseline' or 'resnet'")

    cnn_model = cnn_model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        cnn_model.parameters(), lr=cnn_learning_rate, weight_decay=cnn_weight_decay
    )
    cnn_model, cnn_history = cnn_engine.treinar_modelo(
        model=cnn_model,
        train_loader=cnn_train_loader,
        val_loader=cnn_val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=cnn_epochs,
    )

    cnn_predictions, cnn_targets = cnn_evaluate.collect_predictions(
        cnn_model, cnn_val_loader, device
    )
    cnn_ids = cnn_val_loader.dataset.dataframe.iloc[:, 0].astype(str).to_numpy()
    if not np.array_equal(feature_results["validation_ids"], cnn_ids):
        raise RuntimeError("Validation image IDs are not aligned between CNN and feature models")
    if not np.array_equal(feature_results["validation_targets"], cnn_targets):
        raise RuntimeError("Validation targets are not aligned between CNN and feature models")
    if plot_examples:
        cnn_evaluate.plot_prediction_examples(cnn_val_loader, cnn_predictions, cnn_targets)

    image_cnn_q = calculate_yules_q(
        feature_results["image_predictions"], cnn_predictions, cnn_targets
    )
    fft_cnn_q = calculate_yules_q(
        feature_results["fft_predictions"], cnn_predictions, cnn_targets
    )
    print("\nCNN error comparison")
    _print_yules_q("image statistics vs CNN", image_cnn_q)
    _print_yules_q("FFT vs CNN", fft_cnn_q)

    return {
        **feature_results,
        "cnn_model": cnn_model,
        "cnn_history": cnn_history,
        "cnn_predictions": cnn_predictions,
        "image_cnn_error_association": image_cnn_q,
        "fft_cnn_error_association": fft_cnn_q,
    }


if __name__ == "__main__":
    train_and_compare_all_models(mode="proto", epochs=5)
