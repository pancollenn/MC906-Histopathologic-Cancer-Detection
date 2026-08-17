# 🔬 Histopathologic Cancer Detection — MC906 (IC / UNICAMP)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Histopathologic--Cancer--Detection-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/histopathologic-cancer-detection)
[![Course](https://img.shields.io/badge/UNICAMP-MC906-green.svg)](https://www.ic.unicamp.br/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Projeto desenvolvido para a disciplina **MC906 — Introdução à Inteligência Artificial** do **Instituto de Computação (IC) da Universidade Estadual de Campinas (UNICAMP)**.

O objetivo do projeto é a **identificação automatizada de metástase de câncer em biópsias de linfonodos**, a partir da classificação binária de imagens histopatológicas (*PatchCamelyon benchmark / Kaggle Challenge*).

---

## 📌 Sumário

- [Visão Geral](#-visão-geral)
- [Descrição do Problema & Dataset](#-descrição-do-problema--dataset)
- [Arquitetura & Metodologia](#-arquitetura--metodologia)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Instalação e Configuração](#-instalação-e-configuração)
- [Como Executar](#-como-executar)
- [Resultados e Avaliação](#-resultados-e-avaliação)
- [Interpretabilidade (Grad-CAM)](#-interpretabilidade-grad-cam)
- [Autores & Agradecimentos](#-autores--agradecimentos)
- [Licença](#-licença)

---

## 🩺 Visão Geral

A detecção precoce e precisa de metástases em linfonodos é uma etapa crítica no estadiamento do câncer de mama e no planejamento terapêutico de pacientes oncológicos. No entanto, o exame histopatológico manual de lâminas digitalizadas (*Whole-Slide Images — WSI*) é uma tarefa laboriosa, demorada e suscetível à variabilidade inter e intra-observador.

Este projeto propõe uma abordagem de **Deep Learning / Visão Computacional** baseada em Redes Neurais Convolucionais (CNNs) e Transfer Learning para classificar pequenos fragmentos (*patches*) de tecido como contendo ou não células tumorais malignas.

---

## 📊 Descrição do Problema & Dataset

O projeto utiliza a base do desafio do Kaggle [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection), adaptada do benchmark **PatchCamelyon (PCam)**.

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Dimensões da Imagem</b></td>
      <td align="center">96 × 96 pixels (RGB)</td>
    </tr>
    <tr>
      <td align="center"><b>Região de Decisão (ROI)</b></td>
      <td align="center">Centro de 32 × 32 pixels</td>
    </tr>
    <tr>
      <td align="center"><b>Classes</b></td>
      <td align="center"><code>0</code>: Tecido Saudável / Sem Tumor<br><code>1</code>: Tecido Metastático (≥ 1 pixel tumoral no centro)</td>
    </tr>
    <tr>
      <td align="center"><b>Total de Amostras (Treino)</b></td>
      <td align="center">~220.025 imagens rotuladas</td>
    </tr>
    <tr>
      <td align="center"><b>Métrica Oficial</b></td>
      <td align="center"><b>ROC-AUC</b> (Area Under the ROC Curve)</td>
    </tr>
  </table>
</div>

> ⚠️ **Regra Fundamental do Dataset:** A presença de tecido tumoral na periferia do patch (fora da região central de 32×32 px) **não** afeta o rótulo da imagem. O modelo deve focar estritamente na área central para emitir sua predição.

---

## 🧠 Arquitetura & Metodologia

O pipeline foi estruturado em quatro etapas principais:

```mermaid
flowchart LR
    A[Raw Patches\n96x96 RGB] --> B[Data Augmentation\n& Preprocessing]
    B --> C[Backbone CNN\nResNet / EfficientNet]
    C --> D[Classification Head\nDropout + Dense]
    D --> E[Sigmoid Output\nProbabilidade de Tumor]
    E --> F[ROC-AUC &\nMetrics Evaluation]
