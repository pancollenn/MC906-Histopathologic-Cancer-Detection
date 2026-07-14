"""Train the handcrafted-feature models and CNN on one shared validation split."""

import os

import kagglehub
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from cnn_resnet import data_setup as cnn_data_setup
from cnn_resnet import engine as cnn_engine
from cnn_resnet import evaluate as cnn_evaluate
from cnn_resnet.model_builder import BaselineCNN, get_resnet18_model
from ensemble.train import calculate_yules_q, train_and_compare_models
from models import FFTResNetEnsemble


def _print_yules_q(label, association):
    """Print an error-association result with its contingency-table counts."""
    print(
        f"  - {label}: Q={association['yules_q']:.4f} | "
        f"ambos erraram={association['both_wrong']}, "
        f"somente o primeiro errou={association['first_only_wrong']}, "
        f"somente o segundo errou={association['second_only_wrong']}"
    )


def _format_metric(value):
    """Format a metric for the final report table, including unavailable values."""
    return "—" if value is None else f"{value:.4f}"


def _print_experiment_header(
    mode,
    device,
    batch_size,
    feature_epochs,
    cnn_epochs,
    ensemble_epochs,
    cnn_model_type,
):
    """Print the reproducible settings used by the experiment."""
    cnn_name = "ResNet-18 pré-treinada" if cnn_model_type == "resnet" else "CNN baseline"
    print("\n" + "=" * 88)
    print("EXPERIMENTO: comparação de classificadores para detecção histopatológica")
    print("=" * 88)
    print("Divisão: estratificada 90% treino / 10% validação (semente 42, compartilhada).")
    print(f"Dispositivo: {device} | modo: {mode} | batch size: {batch_size}")
    print(
        "Modelos: estatísticas de imagem, descritores FFT, "
        f"{cnn_name} e ensemble FFT + ResNet."
    )
    print(
        "Épocas — descritores: "
        f"{feature_epochs}; CNN: {cnn_epochs}; ensemble: {ensemble_epochs}."
    )
    print("=" * 88)


def _print_dataset_summary(train_loader, val_loader):
    """Print dataset size and class prevalence, useful when citing an experiment."""
    train_labels = train_loader.dataset.dataframe.iloc[:, 1]
    val_labels = val_loader.dataset.dataframe.iloc[:, 1]
    summary = {
        "training_samples": len(train_labels),
        "validation_samples": len(val_labels),
        "training_tumor_rate": float(train_labels.mean()),
        "validation_tumor_rate": float(val_labels.mean()),
    }
    print("\nDados utilizados")
    print(
        f"  Treino: {summary['training_samples']:,} imagens "
        f"({summary['training_tumor_rate']:.2%} tumor) | "
        f"Validação: {summary['validation_samples']:,} imagens "
        f"({summary['validation_tumor_rate']:.2%} tumor)"
    )
    return summary


def _print_final_metrics(rows):
    """Print a compact, report-ready comparison of final validation metrics."""
    print("\nResumo final na validação")
    print("-" * 112)
    print(
        f"{'Modelo':<31} {'Loss treino':>12} {'Loss val.':>11} "
        f"{'Acurácia':>10} {'Precisão':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}"
    )
    print("-" * 112)
    for name, metrics in rows:
        print(
            f"{name:<31} {_format_metric(metrics.get('train_loss')):>12} "
            f"{_format_metric(metrics.get('val_loss')):>11} "
            f"{_format_metric(metrics.get('accuracy')):>10} "
            f"{_format_metric(metrics.get('precision')):>10} "
            f"{_format_metric(metrics.get('recall')):>10} "
            f"{_format_metric(metrics.get('f1')):>10} "
            f"{_format_metric(metrics.get('auc')):>10}"
        )
    print("-" * 112)
    print("Nota: — indica métrica não calculada pelo respectivo fluxo de treinamento.")


def _classification_metrics(predictions, targets):
    """Calculate the common threshold-based validation metrics for one model."""
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }


class _FFTResNetDataset(Dataset):
    """Pair the aligned FFT descriptors and CNN images for late fusion."""

    def __init__(self, fft_dataset, image_dataset):
        if len(fft_dataset) != len(image_dataset):
            raise ValueError("FFT and CNN datasets must have the same number of samples")
        fft_ids = fft_dataset.dataframe.iloc[:, 0].astype(str).to_numpy()
        image_ids = image_dataset.dataframe.iloc[:, 0].astype(str).to_numpy()
        if not np.array_equal(fft_ids, image_ids):
            raise ValueError("FFT and CNN datasets are not aligned")
        self.fft_dataset = fft_dataset
        self.image_dataset = image_dataset

    def __len__(self):
        return len(self.fft_dataset)

    def __getitem__(self, idx):
        fft_features, fft_label = self.fft_dataset[idx]
        image, image_label = self.image_dataset[idx]
        if fft_label.item() != image_label:
            raise RuntimeError("Mismatched labels in FFT/ResNet ensemble dataset")
        return fft_features, image, fft_label


def _create_ensemble_loaders(fft_train_loader, fft_val_loader, cnn_train_loader, cnn_val_loader, batch_size, num_workers):
    """Build paired loaders after the base models have been trained separately."""
    options = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        # Preserve each worker's FFT-feature cache across ensemble epochs.
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(
            _FFTResNetDataset(fft_train_loader.dataset, cnn_train_loader.dataset),
            batch_size=batch_size,
            shuffle=True,
            **options,
        ),
        DataLoader(
            _FFTResNetDataset(fft_val_loader.dataset, cnn_val_loader.dataset),
            batch_size=batch_size,
            shuffle=False,
            **options,
        ),
    )


def _evaluate_ensemble(model, data_loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    predictions, targets = [], []
    with torch.no_grad():
        for fft_features, images, labels in data_loader:
            fft_features = fft_features.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)
            logits = model(fft_features, images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            predictions.extend((torch.sigmoid(logits) >= 0.5).long().cpu().view(-1).tolist())
            targets.extend(labels.long().cpu().view(-1).tolist())

    return {
        "loss": loss_sum / len(data_loader.dataset),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "predictions": np.asarray(predictions),
        "targets": np.asarray(targets),
    }


def _train_ensemble(model, train_loader, val_loader, device, epochs, learning_rate, weight_decay):
    """Train only the ensemble fusion head; FFT and ResNet stay frozen."""
    criterion = nn.BCEWithLogitsLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    history = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for fft_features, images, labels in train_loader:
            fft_features = fft_features.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(fft_features, images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)

        metrics = _evaluate_ensemble(model, val_loader, criterion, device)
        metrics["train_loss"] = loss_sum / len(train_loader.dataset)
        history.append(metrics)
        print(
            f"  Ensemble — época {epoch + 1:02d}/{epochs}: "
            f"loss de treino={metrics['train_loss']:.4f} | "
            f"loss de validação={metrics['loss']:.4f} | "
            f"acurácia de validação={metrics['accuracy']:.4f}"
        )
    return model, history


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
    cnn_num_workers=None,
    ensemble_epochs=None,
    ensemble_learning_rate=1e-3,
    ensemble_weight_decay=1e-6,
    ensemble_hidden_dim=16,
    plot_examples=False,
    device=None,
):
    """Train separate FFT/ResNet models, then train their frozen ensemble.

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
    if ensemble_epochs is None:
        ensemble_epochs = epochs
    if cnn_model_type not in {"baseline", "resnet"}:
        raise ValueError("cnn_model_type must be 'baseline' or 'resnet'")
    if cnn_num_workers is None:
        cnn_num_workers = os.cpu_count() or 1

    _print_experiment_header(
        mode,
        device,
        batch_size,
        epochs,
        cnn_epochs,
        ensemble_epochs,
        cnn_model_type,
    )

    print("\n[Etapa 1/3] Treinando os classificadores de descritores manualmente extraídos")
    print("  - Modelo de estatísticas de imagem e modelo de descritores FFT.")
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
    dataset_summary = _print_dataset_summary(cnn_train_loader, cnn_val_loader)
    if cnn_model_type == "baseline":
        cnn_model = BaselineCNN()
    elif cnn_model_type == "resnet":
        cnn_model = get_resnet18_model(pretrained=True, fine_tune=True)

    cnn_model = cnn_model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        cnn_model.parameters(), lr=cnn_learning_rate, weight_decay=cnn_weight_decay
    )
    print("\n[Etapa 2/3] Treinando a rede convolucional independentemente")
    print(
        f"  - Arquitetura: {'ResNet-18 pré-treinada no ImageNet' if cnn_model_type == 'resnet' else 'CNN baseline'}. "
        f"Taxa de aprendizado: {cnn_learning_rate:g}; weight decay: {cnn_weight_decay:g}."
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

    # The base branches have now been trained independently.  Pair their same
    # train/validation split and optimize only the fusion head.
    ensemble_train_loader, ensemble_val_loader = _create_ensemble_loaders(
        feature_results["fft_train_loader"],
        feature_results["fft_val_loader"],
        cnn_train_loader,
        cnn_val_loader,
        batch_size,
        cnn_num_workers,
    )
    fft_resnet_ensemble = FFTResNetEnsemble(
        feature_results["fft_model"], cnn_model, hidden_dim=ensemble_hidden_dim
    ).to(device)
    frozen_parameters = sum(
        parameter.numel()
        for backbone in (fft_resnet_ensemble.fft_model, fft_resnet_ensemble.resnet_model)
        for parameter in backbone.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in fft_resnet_ensemble.parameters()
        if parameter.requires_grad
    )
    print("\n[Etapa 3/3] Treinando o ensemble de fusão tardia FFT + ResNet")
    print(
        f"  - Bases congeladas: {frozen_parameters:,} parâmetros (FFT e ResNet). "
        f"Cabeça de fusão treinável: {trainable_parameters:,} parâmetros."
    )
    print(
        f"  - Cabeça: 2 logits → camada oculta de {ensemble_hidden_dim} neurônios → 1 logit. "
        f"Taxa de aprendizado: {ensemble_learning_rate:g}; weight decay: {ensemble_weight_decay:g}."
    )
    fft_resnet_ensemble, ensemble_history = _train_ensemble(
        fft_resnet_ensemble,
        ensemble_train_loader,
        ensemble_val_loader,
        device,
        ensemble_epochs,
        ensemble_learning_rate,
        ensemble_weight_decay,
    )
    ensemble_metrics = ensemble_history[-1] if ensemble_history else {}

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
    ensemble_predictions = ensemble_metrics.get("predictions", np.array([]))
    ensemble_targets = ensemble_metrics.get("targets", np.array([]))
    if not np.array_equal(ensemble_targets, cnn_targets):
        raise RuntimeError("Validation targets are not aligned with the ensemble")
    ensemble_cnn_q = calculate_yules_q(ensemble_predictions, cnn_predictions, cnn_targets)
    cnn_metrics = {
        **_classification_metrics(cnn_predictions, cnn_targets),
        "train_loss": cnn_history["train_loss"][-1] if cnn_history["train_loss"] else None,
        "val_loss": cnn_history["val_loss"][-1] if cnn_history["val_loss"] else None,
        "auc": cnn_history["val_auc"][-1] if cnn_history["val_auc"] else None,
    }
    ensemble_report_metrics = {
        **ensemble_metrics,
        "val_loss": ensemble_metrics.get("loss"),
    }
    final_metric_rows = [
        ("Estatísticas de imagem", feature_results["image_metrics"]),
        ("Descritores FFT", feature_results["fft_metrics"]),
        (
            "ResNet-18" if cnn_model_type == "resnet" else "CNN baseline",
            cnn_metrics,
        ),
        ("Ensemble FFT + ResNet", ensemble_report_metrics),
    ]
    _print_final_metrics(final_metric_rows)
    print("\nAssociação entre erros na validação (Yule's Q)")
    print("  Q próximo de 0 sugere erros pouco associados; Q negativo sugere complementaridade.")
    _print_yules_q("Estatísticas de imagem × FFT", feature_results["error_association"])
    _print_yules_q("Estatísticas de imagem × CNN", image_cnn_q)
    _print_yules_q("FFT × CNN", fft_cnn_q)
    _print_yules_q("Ensemble FFT + ResNet × CNN", ensemble_cnn_q)
    print("\nExperimento concluído. As métricas acima referem-se exclusivamente à divisão de validação.")

    report = {
        "configuration": {
            "mode": mode,
            "seed": 42,
            "batch_size": batch_size,
            "feature_epochs": epochs,
            "cnn_epochs": cnn_epochs,
            "ensemble_epochs": ensemble_epochs,
            "cnn_model_type": cnn_model_type,
            "ensemble_hidden_dim": ensemble_hidden_dim,
        },
        "dataset": dataset_summary,
        "validation_metrics": dict(final_metric_rows),
        "error_associations": {
            "image_statistics_vs_fft": feature_results["error_association"],
            "image_statistics_vs_cnn": image_cnn_q,
            "fft_vs_cnn": fft_cnn_q,
            "ensemble_vs_cnn": ensemble_cnn_q,
        },
    }

    return {
        **feature_results,
        "cnn_model": cnn_model,
        "cnn_history": cnn_history,
        "cnn_metrics": cnn_metrics,
        "cnn_predictions": cnn_predictions,
        "fft_resnet_ensemble": fft_resnet_ensemble,
        "ensemble_history": ensemble_history,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_predictions": ensemble_predictions,
        "report": report,
        "image_cnn_error_association": image_cnn_q,
        "fft_cnn_error_association": fft_cnn_q,
        "ensemble_cnn_error_association": ensemble_cnn_q,
    }


if __name__ == "__main__":
    train_and_compare_all_models(mode="proto", epochs=5, cnn_epochs=5, ensemble_epochs=5)
