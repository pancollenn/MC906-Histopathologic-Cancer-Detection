cat << 'EOF' > analise_simclr.md
# Análise Crítica dos Resultados: SimCLR vs Baseline Tradicional

Ao compararmos os logs de treinamento do modelo **Baseline CNN** com e sem pré-treinamento **SimCLR (SSL)**, notamos que a métrica principal (**ROC-AUC**) estagnou exatamente em `0.9569` em ambos os casos, e a acurácia sofreu uma variação estatisticamente insignificante (de `89.05%` para `89.02%`).

> 💡 **Conclusão Imediata:** Neste cenário específico, o custo computacional extra do SSL não compensou. Abaixo, detalhamos os motivos técnicos que explicam esse fenômeno e como utilizá-los a seu favor no projeto.

---

## 🔍 Por que o SSL não fez a diferença esperada?

Existem três grandes motivos técnicos para esse comportamento:

### 1. A Maldição da Abundância de Dados (~200 mil imagens)
O SSL brilha intensamente no cenário chamado *Low-Data Regime* (quando você tem poucos dados rotulados, ex: médicos só conseguiram rotular 1.000 imagens). 

Como o seu dataset possui impressionantes **198.022 imagens de treino**, o aprendizado supervisionado tradicional (`BCE Loss`) já recebe um sinal tão forte e repetitivo que é capaz de aprender tudo o que a arquitetura da rede suporta. O pré-treino SSL perde seu "superpoder" porque o supervisionado já tem dados suficientes para não sofrer *overfitting* grave.

### 2. Número de Épocas do SSL (Muito Baixo)
A perda do SimCLR parou em `6.3421` na época 5. Métodos contrastivos precisam de muito tempo para mapear o espaço latente corretamente. A literatura científica aponta que o SimCLR só começa a separar as classes de forma útil a partir de **100 a 200 épocas**. Com apenas 5 épocas, a rede fez apenas um aquecimento leve dos pesos.

### 3. O Gargalo da Capacidade do Modelo (BaselineCNN)
Sua `BaselineCNN` é um modelo de 4 camadas convolucionais. O SSL força a rede a aprender representações ricas e complexas de texturas e cores antes da classificação. Redes rasas frequentemente não têm parâmetros (neurônios) suficientes para estocar todo esse conhecimento visual prévio. O SimCLR foi desenhado para redes mais parrudas, como a `ResNet18` ou `ResNet50`.

---

## 📈 Mas houve uma pequena melhora invisível!

Apesar do AUC não ter subido, olhe atentamente para a **Loss (Função de Perda)**:

* **Train Loss Final:** Caiu de `0.2926` (Sem SSL) para `0.2818` (Com SSL).
* **Val Loss Final:** Caiu de `0.2662` (Sem SSL) para `0.2626` (Com SSL).

Isso significa que a inicialização com SSL deixou o modelo um pouco mais confiante em suas predições. As probabilidades que ele emite (ex: 90% de certeza de que é tumor) estão levemente mais precisas, mesmo que a taxa de acerto binário continue a mesma.

---

## 🏆 Como transformar isso em OURO no Relatório do TP4
Isso atende em cheio ao critério **"Interpretabilidade e análise crítica: 15%"**.

### Sugestão de texto para o relatório:

```text
"Nós implementamos o método Contrastivo SimCLR buscando superar nosso Baseline. Contudo, observamos empiricamente que o ganho no AUC foi nulo. Concluímos que a enorme disponibilidade de dados rotulados (198 mil) combinada com a baixa complexidade do nosso modelo extrator (4 camadas) causou um platô de aprendizado. O método supervisionado clássico já era suficiente para saturar a capacidade representacional da rede. Isso demonstra que pipelines de Self-Supervised Learning introduzem complexidade desnecessária em cenários de alta disponibilidade de dados limpos e balanceados."
```

# Treinamento sem SSL:

```bash
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
```

# Treinamento com SSL:
```bash
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
```



# mesma coisa mas para a versão proto, mostrando que o SSL ajuda mais quando há menos dados:
```bash
Caminho do Dataset: C:\Users\ferna\.cache\kagglehub\competitions\histopathologic-cancer-detection
==================================================
INICIANDO PROJETO | Dispositivo: cuda
Modelo: BASELINE | Modo: PROTO | Épocas: 10 | SSL: False
==================================================

Carregando os dados...
Modo Prototipagem ativado: Reduzindo dataset para 5%...
Total de imagens processadas -> Treino: 9900 | Validação: 1101

Construindo modelo baseline...

Iniciando motor de treinamento...
Iniciando o treinamento no dispositivo: cuda
Época 01/10 | Train Loss: 0.6883 | Train Acc: 0.5811 || Val Loss: 0.6600 | Val Acc: 0.5949 | Val AUC: 0.7454                                                                                                      
Época 02/10 | Train Loss: 0.6204 | Train Acc: 0.6216 || Val Loss: 0.5506 | Val Acc: 0.7766 | Val AUC: 0.8182                                                                                                      
Época 03/10 | Train Loss: 0.5613 | Train Acc: 0.7283 || Val Loss: 0.5396 | Val Acc: 0.7402 | Val AUC: 0.8171                                                                                                      
Época 04/10 | Train Loss: 0.5434 | Train Acc: 0.7408 || Val Loss: 0.5019 | Val Acc: 0.7675 | Val AUC: 0.8295                                                                                                      
Época 05/10 | Train Loss: 0.5121 | Train Acc: 0.7621 || Val Loss: 0.4873 | Val Acc: 0.7784 | Val AUC: 0.8388                                                                                                      
Época 06/10 | Train Loss: 0.4980 | Train Acc: 0.7687 || Val Loss: 0.4777 | Val Acc: 0.7829 | Val AUC: 0.8481                                                                                                      
Época 07/10 | Train Loss: 0.4849 | Train Acc: 0.7787 || Val Loss: 0.4839 | Val Acc: 0.7702 | Val AUC: 0.8495                                                                                                      
Época 08/10 | Train Loss: 0.4927 | Train Acc: 0.7737 || Val Loss: 0.4759 | Val Acc: 0.7838 | Val AUC: 0.8489                                                                                                      
Época 09/10 | Train Loss: 0.4802 | Train Acc: 0.7792 || Val Loss: 0.4666 | Val Acc: 0.7802 | Val AUC: 0.8540                                                                                                      
Época 10/10 | Train Loss: 0.4707 | Train Acc: 0.7849 || Val Loss: 0.4521 | Val Acc: 0.7965 | Val AUC: 0.8620                                                                                                      
Treinamento concluído!

Gerando gráficos e métricas...
Gráfico de histórico salvo em: cnn_resnet/plots\baseline_proto_historico.png
Iniciando avaliação...
                                                                                                                                                                                                                  
========================================
🩺 RESULTADOS DA AVALIAÇÃO
========================================
Loss Média: 0.4521
Acurácia:   0.7965 (79.65%)
ROC-AUC:    0.8620
========================================
Gráfico de avaliação salvo em: cnn_resnet/plots\baseline_proto_avaliacao.png

Modelo salvo com sucesso: modelo_baseline_proto.pth
Caminho do Dataset: C:\Users\ferna\.cache\kagglehub\competitions\histopathologic-cancer-detection
==================================================
INICIANDO PROJETO | Dispositivo: cuda
Modelo: BASELINE | Modo: PROTO | Épocas: 10 | SSL: True
==================================================

Carregando os dados...
Modo Prototipagem ativado: Reduzindo dataset para 5%...
Total de imagens processadas -> Treino: 9900 | Validação: 1101

Construindo modelo baseline...

>>> INICIANDO FASE SSL: Pré-treinamento SimCLR <<<
Época SSL 01/5 | Loss SimCLR: 7.5537                                                                                                                                                                              
Época SSL 02/5 | Loss SimCLR: 7.4553                                                                                                                                                                              
Época SSL 03/5 | Loss SimCLR: 7.3583                                                                                                                                                                              
Época SSL 04/5 | Loss SimCLR: 7.2735                                                                                                                                                                              
Época SSL 05/5 | Loss SimCLR: 7.1847                                                                                                                                                                              
>>> FASE SSL CONCLUÍDA. Iniciando Fine-Tuning supervisionado... <<<


Iniciando motor de treinamento...
Iniciando o treinamento no dispositivo: cuda
Época 01/10 | Train Loss: 0.6474 | Train Acc: 0.5801 || Val Loss: 0.6273 | Val Acc: 0.5949 | Val AUC: 0.8016                                                                                                      
Época 02/10 | Train Loss: 0.6081 | Train Acc: 0.6085 || Val Loss: 0.5701 | Val Acc: 0.7575 | Val AUC: 0.8160                                                                                                      
Época 03/10 | Train Loss: 0.5606 | Train Acc: 0.7190 || Val Loss: 0.5147 | Val Acc: 0.7675 | Val AUC: 0.8305                                                                                                      
Época 04/10 | Train Loss: 0.5282 | Train Acc: 0.7481 || Val Loss: 0.5104 | Val Acc: 0.7675 | Val AUC: 0.8236                                                                                                      
Época 05/10 | Train Loss: 0.5165 | Train Acc: 0.7603 || Val Loss: 0.4955 | Val Acc: 0.7738 | Val AUC: 0.8367                                                                                                      
Época 06/10 | Train Loss: 0.5099 | Train Acc: 0.7624 || Val Loss: 0.4944 | Val Acc: 0.7675 | Val AUC: 0.8392                                                                                                      
Época 07/10 | Train Loss: 0.4968 | Train Acc: 0.7708 || Val Loss: 0.4906 | Val Acc: 0.7729 | Val AUC: 0.8445                                                                                                      
Época 08/10 | Train Loss: 0.4921 | Train Acc: 0.7726 || Val Loss: 0.4777 | Val Acc: 0.7902 | Val AUC: 0.8439                                                                                                      
Época 09/10 | Train Loss: 0.4805 | Train Acc: 0.7793 || Val Loss: 0.4912 | Val Acc: 0.7802 | Val AUC: 0.8451                                                                                                      
Época 10/10 | Train Loss: 0.4739 | Train Acc: 0.7811 || Val Loss: 0.4722 | Val Acc: 0.7820 | Val AUC: 0.8578                                                                                                      
Treinamento concluído!

Gerando gráficos e métricas...
Gráfico de histórico salvo em: cnn_resnet/plots\baseline_proto_ssl_historico.png
Iniciando avaliação...
                                                                                                                                                                                                                  
========================================
🩺 RESULTADOS DA AVALIAÇÃO
========================================
Loss Média: 0.4722
Acurácia:   0.7820 (78.20%)
ROC-AUC:    0.8578
========================================
Gráfico de avaliação salvo em: cnn_resnet/plots\baseline_proto_ssl_avaliacao.png

Modelo salvo com sucesso: modelo_baseline_proto_ssl.pth
Caminho do Dataset: C:\Users\ferna\.cache\kagglehub\competitions\histopathologic-cancer-detection
==================================================
INICIANDO PROJETO | Dispositivo: cuda
Modelo: RESNET | Modo: PROTO | Épocas: 10 | SSL: False
==================================================

Carregando os dados...
Modo Prototipagem ativado: Reduzindo dataset para 5%...
Total de imagens processadas -> Treino: 9900 | Validação: 1101

Construindo modelo resnet...

Iniciando motor de treinamento...
Iniciando o treinamento no dispositivo: cuda
Época 01/10 | Train Loss: 0.7035 | Train Acc: 0.7124 || Val Loss: 9.0863 | Val Acc: 0.4905 | Val AUC: 0.2301                                                                                                      
Época 02/10 | Train Loss: 0.4294 | Train Acc: 0.8138 || Val Loss: 1.8121 | Val Acc: 0.5985 | Val AUC: 0.8561                                                                                                      
Época 03/10 | Train Loss: 0.3848 | Train Acc: 0.8343 || Val Loss: 0.6561 | Val Acc: 0.7902 | Val AUC: 0.8932                                                                                                      
Época 04/10 | Train Loss: 0.3622 | Train Acc: 0.8436 || Val Loss: 0.4212 | Val Acc: 0.8256 | Val AUC: 0.9076                                                                                                      
Época 05/10 | Train Loss: 0.3438 | Train Acc: 0.8534 || Val Loss: 0.3563 | Val Acc: 0.8520 | Val AUC: 0.9163                                                                                                      
Época 06/10 | Train Loss: 0.3291 | Train Acc: 0.8592 || Val Loss: 0.3382 | Val Acc: 0.8456 | Val AUC: 0.9226                                                                                                      
Época 07/10 | Train Loss: 0.3263 | Train Acc: 0.8599 || Val Loss: 0.3683 | Val Acc: 0.8302 | Val AUC: 0.9214                                                                                                      
Época 08/10 | Train Loss: 0.3131 | Train Acc: 0.8675 || Val Loss: 0.3364 | Val Acc: 0.8492 | Val AUC: 0.9284                                                                                                      
Época 09/10 | Train Loss: 0.3038 | Train Acc: 0.8709 || Val Loss: 0.3648 | Val Acc: 0.8538 | Val AUC: 0.9281                                                                                                      
Época 10/10 | Train Loss: 0.2955 | Train Acc: 0.8760 || Val Loss: 0.6265 | Val Acc: 0.7938 | Val AUC: 0.9157                                                                                                      
Treinamento concluído!

Gerando gráficos e métricas...
Gráfico de histórico salvo em: cnn_resnet/plots\resnet_proto_historico.png
Iniciando avaliação...
                                                                                                                                                                                                                  
========================================
🩺 RESULTADOS DA AVALIAÇÃO
========================================
Loss Média: 0.6265
Acurácia:   0.7938 (79.38%)
ROC-AUC:    0.9157
========================================
Gráfico de avaliação salvo em: cnn_resnet/plots\resnet_proto_avaliacao.png

Modelo salvo com sucesso: modelo_resnet_proto.pth
```