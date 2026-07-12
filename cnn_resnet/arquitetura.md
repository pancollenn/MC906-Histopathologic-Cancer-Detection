# Arquitetura do Projeto

O projeto segue as melhores práticas da engenharia de **Machine Learning**, organizando o código em módulos bem definidos, em vez de concentrar toda a lógica em um único Jupyter Notebook. Cada arquivo possui uma responsabilidade específica, tornando o código mais organizado, reutilizável e fácil de manter.

---

## 📦 `data_setup.py` — Processamento de Dados

Responsável por localizar as imagens, carregar as tabelas e preparar os dados para treinamento da rede neural.

### Principais componentes

- **`set_seed()`**
  - Define uma semente fixa para os geradores de números aleatórios do:
    - Python
    - NumPy
    - PyTorch
  - Garante que os experimentos sejam **100% reproduzíveis**.

- **`HistopathologicDataset`**
  - Classe personalizada (`Dataset`) do PyTorch.
  - Lê as imagens `.tif` utilizando o arquivo CSV do Kaggle.
  - Converte automaticamente as imagens para o formato **RGB**.

- **`create_dataloaders()`**
  - Divide o conjunto de dados em:
    - **90%** para treinamento
    - **10%** para validação
  - Aplica técnicas de **Data Augmentation**, como:
    - rotações aleatórias;
    - cortes aleatórios (*Random Crop*);
    - outras transformações para evitar **overfitting**.
  - Normaliza os pixels utilizando o padrão do **ImageNet**.
  - Cria e retorna os objetos `DataLoader` utilizados durante o treinamento.

---

## 🧠 `model_builder.py` — Construção dos Modelos

Arquivo responsável por criar ou importar as arquiteturas de redes neurais.

### Modelos disponíveis

### `BaselineCNN`

Rede neural convolucional construída do zero, composta por:

- 4 blocos convolucionais (`Conv2d`);
- camadas de redução espacial (`MaxPool2d`);
- redução progressiva da imagem:

```
64 × 64
   ↓
32 × 32
   ↓
16 × 16
   ↓
8 × 8
   ↓
4 × 4
```

Após a extração das características, utiliza camadas totalmente conectadas (*Linear*) para realizar a classificação final:

- **0 → Normal**
- **1 → Tumor**

### `get_resnet18_model()`

Importa a arquitetura **ResNet18** da biblioteca `torchvision`, permitindo utilizar:

- **Transfer Learning**
  - aproveitando o conhecimento adquirido em milhões de imagens do ImageNet;

- **Fine-Tuning**
  - adaptando a rede ao problema de detecção de câncer histopatológico.

---

## ⚙️ `engine.py` — Motor de Treinamento

Contém a função responsável pelo treinamento do modelo.

### `treinar_modelo()`

Executa o ciclo completo de treinamento ao longo das épocas.

Em cada época são realizadas duas etapas:

### 🔹 Fase de Treinamento

Durante essa etapa:

1. o modelo realiza previsões;
2. a função de perda (*Loss*) é calculada;
3. os gradientes são propagados utilizando:

```python
loss.backward()
```

4. o otimizador atualiza os pesos por meio de:

```python
optimizer.step()
```

### 🔹 Fase de Validação

Após o treinamento da época:

- o modelo é avaliado utilizando imagens que **não participaram do treinamento**;
- os pesos permanecem inalterados;
- mede-se a capacidade de generalização da rede.

### Métricas calculadas

- Loss
- Accuracy
- ROC-AUC

---

## 📊 `evaluate.py` — Avaliação e Visualização

Responsável pela análise visual e estatística do desempenho do modelo.

### Principais funções

### `visualizar_batch()`

- Seleciona um lote aleatório de imagens;
- desfaz a normalização aplicada anteriormente;
- gera uma grade contendo:
  - as imagens;
  - seus respectivos rótulos.

Essa visualização permite verificar se o **Data Augmentation** está sendo aplicado corretamente.

### `plotar_historico()`

Gera gráficos mostrando a evolução das métricas durante o treinamento, como:

- Loss
- Accuracy
- ROC-AUC

### `avaliar_modelo()`

Produz e salva:

- Matriz de Confusão;
- Curva ROC;
- demais métricas finais do modelo.

Todos os resultados são armazenados na pasta do projeto.

---

## 🚀 `main.py` / `main.ipynb` — Orquestrador do Projeto

É o ponto de entrada da aplicação, responsável por coordenar todos os módulos.

### Fluxo de execução

1. Baixa automaticamente o dataset utilizando **kagglehub**;
2. Define os hiperparâmetros do treinamento:
   - número de épocas;
   - tamanho do *batch*;
   - taxa de aprendizado;
3. Seleciona automaticamente o dispositivo de execução:
   - CPU;
   - GPU;
4. Inicia o treinamento chamando o módulo `engine.py`;
5. Executa a avaliação do modelo utilizando `evaluate.py`;
6. Salva os pesos finais da rede neural em um arquivo `.pth`, permitindo reutilizar o modelo treinado posteriormente.

---

# Resumo da Arquitetura

| Módulo | Responsabilidade |
|---------|------------------|
| **`data_setup.py`** | Carregamento, processamento e preparação dos dados |
| **`model_builder.py`** | Construção e importação das arquiteturas de redes neurais |
| **`engine.py`** | Execução do treinamento e validação do modelo |
| **`evaluate.py`** | Avaliação estatística e geração de gráficos |
| **`main.py` / `main.ipynb`** | Coordenação de todo o fluxo de treinamento e salvamento do modelo |



Treinamento sem SSL:
$ python main.py 
Caminho do Dataset: C:\Users\ferna\.cache\kagglehub\competitions\histopathologic-cancer-detection
==================================================
INICIANDO PROJETO | Dispositivo: cuda
Modelo: BASELINE | Modo: FULL | Épocas: 10
==================================================

Carregando os dados...
Total de imagens processadas -> Treino: 198022 | Validação: 22003

Construindo modelo baseline...

Iniciando motor de treinamento...
Iniciando o treinamento no dispositivo: cuda
Época 01/10 | Train Loss: 0.4948 | Train Acc: 0.7606 || Val Loss: 0.4233 | Val Acc: 0.8034 | Val AUC: 0.8872
Época 02/10 | Train Loss: 0.4177 | Train Acc: 0.8130 || Val Loss: 0.3937 | Val Acc: 0.8283 | Val AUC: 0.9041
Época 03/10 | Train Loss: 0.3778 | Train Acc: 0.8341 || Val Loss: 0.3650 | Val Acc: 0.8363 | Val AUC: 0.9237
Época 04/10 | Train Loss: 0.3473 | Train Acc: 0.8494 || Val Loss: 0.3033 | Val Acc: 0.8699 | Val AUC: 0.9405
Época 05/10 | Train Loss: 0.3330 | Train Acc: 0.8572 || Val Loss: 0.3130 | Val Acc: 0.8657 | Val AUC: 0.9450
Época 06/10 | Train Loss: 0.3225 | Train Acc: 0.8620 || Val Loss: 0.2854 | Val Acc: 0.8809 | Val AUC: 0.9486
Época 07/10 | Train Loss: 0.3095 | Train Acc: 0.8700 || Val Loss: 0.2858 | Val Acc: 0.8788 | Val AUC: 0.9481
Época 08/10 | Train Loss: 0.3087 | Train Acc: 0.8703 || Val Loss: 0.2745 | Val Acc: 0.8873 | Val AUC: 0.9538
Época 09/10 | Train Loss: 0.2953 | Train Acc: 0.8774 || Val Loss: 0.2616 | Val Acc: 0.8931 | Val AUC: 0.9583
Época 10/10 | Train Loss: 0.2926 | Train Acc: 0.8786 || Val Loss: 0.2662 | Val Acc: 0.8905 | Val AUC: 0.9569
Treinamento concluído!

Gerando gráficos e métricas...
Gráfico de histórico salvo em: cnn_resnet/plots\baseline_full_historico.png
Iniciando avaliação...
                                                                                                         
========================================
🩺 RESULTADOS DA AVALIAÇÃO
========================================
Loss Média: 0.2662
Acurácia:   0.8905 (89.05%)
ROC-AUC:    0.9569
========================================
Gráfico de avaliação salvo em: cnn_resnet/plots\baseline_full_avaliacao.png

Modelo salvo com sucesso: modelo_baseline_full.pth



Treinamento com SSL:
$ python main.py
Caminho do Dataset: C:\Users\ferna\.cache\kagglehub\competitions\histopathologic-cancer-detection
==================================================
INICIANDO PROJETO | Dispositivo: cuda
Modelo: BASELINE | Modo: FULL | Épocas: 10 | SSL: True
==================================================

Carregando os dados...
Total de imagens processadas -> Treino: 198022 | Validação: 22003

Construindo modelo baseline...

>>> INICIANDO FASE SSL: Pré-treinamento SimCLR <<<
Época SSL 01/5 | Loss SimCLR: 6.9505                                                                                                                                                                              
Época SSL 02/5 | Loss SimCLR: 6.5630                                                                                                                                                                              
Época SSL 03/5 | Loss SimCLR: 6.4519                                                                                                                                                                              
Época SSL 04/5 | Loss SimCLR: 6.3876                                                                                                                                                                              
Época SSL 05/5 | Loss SimCLR: 6.3421                                                                                                                                                                              
>>> FASE SSL CONCLUÍDA. Iniciando Fine-Tuning supervisionado... <<<


Iniciando motor de treinamento...
Iniciando o treinamento no dispositivo: cuda
Época 01/10 | Train Loss: 0.4715 | Train Acc: 0.7800 || Val Loss: 0.4191 | Val Acc: 0.8060 | Val AUC: 0.8947                                                                                                      
Época 02/10 | Train Loss: 0.3988 | Train Acc: 0.8240 || Val Loss: 0.3482 | Val Acc: 0.8479 | Val AUC: 0.9235                                                                                                      
Época 03/10 | Train Loss: 0.3540 | Train Acc: 0.8475 || Val Loss: 0.3278 | Val Acc: 0.8587 | Val AUC: 0.9320                                                                                                      
Época 04/10 | Train Loss: 0.3400 | Train Acc: 0.8546 || Val Loss: 0.3070 | Val Acc: 0.8720 | Val AUC: 0.9415                                                                                                      
Época 05/10 | Train Loss: 0.3243 | Train Acc: 0.8638 || Val Loss: 0.2819 | Val Acc: 0.8836 | Val AUC: 0.9494                                                                                                      
Época 06/10 | Train Loss: 0.3104 | Train Acc: 0.8705 || Val Loss: 0.2772 | Val Acc: 0.8861 | Val AUC: 0.9525                                                                                                      
Época 07/10 | Train Loss: 0.3054 | Train Acc: 0.8731 || Val Loss: 0.2883 | Val Acc: 0.8760 | Val AUC: 0.9499                                                                                                      
Época 08/10 | Train Loss: 0.2974 | Train Acc: 0.8769 || Val Loss: 0.2710 | Val Acc: 0.8887 | Val AUC: 0.9537                                                                                                      
Época 09/10 | Train Loss: 0.2889 | Train Acc: 0.8812 || Val Loss: 0.2500 | Val Acc: 0.8980 | Val AUC: 0.9595                                                                                                      
Época 10/10 | Train Loss: 0.2818 | Train Acc: 0.8844 || Val Loss: 0.2626 | Val Acc: 0.8902 | Val AUC: 0.9569                                                                                                      
Treinamento concluído!

Gerando gráficos e métricas...
Gráfico de histórico salvo em: cnn_resnet/plots\baseline_full_ssl_historico.png
Iniciando avaliação...
                                                                                                                                                                                                                  
========================================
🩺 RESULTADOS DA AVALIAÇÃO
========================================
Loss Média: 0.2626
Acurácia:   0.8902 (89.02%)
ROC-AUC:    0.9569
========================================
Gráfico de avaliação salvo em: cnn_resnet/plots\baseline_full_ssl_avaliacao.png

Modelo salvo com sucesso: modelo_baseline_full_ssl.pth
(mc906_env) 
ferna@DESKTOP-32HTJIA MINGW64 /c/UNICAMP/7_Semestre/MC906-Histopathologic-Cancer-Detection/cnn_resnet (fernando/self-supervised-learning)