 Gemini
Nova conversa
Pesquisar conversas
Estudantes
Imagens
Vídeos
Biblioteca
Novo notebook
Deep Learning: Recurrent Neural Networks and Sequential Data Modeling
Fundamentals of Statistical Machine Learning and Model Assessment
Projeto Oídio para Python
Roteiro Full-Stack: Front, Back e Git
Criação de README para Projeto de IA
README Generation for Cancer Detection
Currículo de Estágio em Tecnologia
Vagas Estágio TI Campinas/SP 2027
Ideias Fáceis para Festa a Fantasia
Descrições para Comissão de Formatura
Alterar Cores no Canva e Canvas
Gasto Calórico: Correr vs. Caminhar
Gasto Calórico: Corrida vs. Caminhada
Sintomas e Padrão da Asma por Esforço
Conteúdo da Disciplina Redes de Computadores
Concursos Públicos de TI: Guia Completo
Queda no Balde, Observação Continua
Correção de Viés: Variável Primária vs. Derivada
Prevenção ao Suicídio e Apoio
Nota de Treineiro no Enem: Uso e Possibilidades
CRI e CRA: BTG e Nubank
Amido Resistente: Arroz e Geladeira
Simulação Investimento CDI 100 Mil
Informações sobre Regiane Erika Koyama
Undervolt Ryzen: Limites e Riscos
Mira Branca em Jogos: Soluções
Cálculo de Exposição ao Risco Bancário
Importância e Uso de Serviços de Mensageria AWS
Atualizar branch com merge ou rebase
MLE: Risco e Mercado de Energia
Btrfs: Sistema de Arquivos Detalhado
Atualizar Pull Request com Alteração Faltante
Consultando Dados de Duas Tabelas
Renomear Fila SQS: Processo de Substituição
Ryzen 5 8600G: Valorant no Mínimo
Mensagem de Estágio no LinkedIn
Slides Para Apresentação de Projeto
Como criar um serviço que captura dados e publica para um SNS da aws em c#
Presentes para estudante de Administração
Conversa com o Gemini

Quero que voce crie um README.md para meu projeto, sem criar dados apenas utilizando os dados e arquivos do projeto e do relatorio. https://github.com/pancollenn/MC906-Histopathologic-Cancer-Detection/

Your Markdown file is ready
Ícone do app MD
README
 MD 

O README foi estruturado utilizando exclusivamente as informações extraídas do relatório "MC906Trabalho4.pdf". Ele inclui os detalhes da equipe, os métodos de pré-processamento, os modelos avaliados (Baseline, ResNet-18, ViT, SimCLR e Ensemble), as métricas de sucesso descritas nas tabelas e as limitações/trabalhos futuros mencionados ao final do documento. Nenhuma informação extra ou fictícia foi adicionada.  


O Gemini é uma IA e pode cometer erros.
Outlining the Project Goals

# MC906 - Histopathologic Cancer Detection

**Universidade Estadual de Campinas (UNICAMP)**  
**Instituto de Computação**  
**Disciplina:** MC906 - Introdução à Inteligência Artificial (Trabalho 4 - Julho 2026)

Repositório do projeto: [https://github.com/pancollenn/MC906-Histopathologic-Cancer-Detection/](https://github.com/pancollenn/MC906-Histopathologic-Cancer-Detection/)

---

## 👥 Equipe
- Bruno Jambeiro Mesquita (RA: 260382)
- Lucas Rodrigues de Mendonça (RA: 236800)
- Fernando Rodrigues da Silva (RA: 247409)
- Thiago Augusto de Tulio Nascimento (RA: 252937)
- Victor Itiro Ogitsu (RA: 244075)

## 🔬 Sobre o Projeto
Este projeto tem como objetivo solucionar o desafio de classificação binária da competição *Histopathologic Cancer Detection* do Kaggle. O foco é identificar a presença de tecido metastático na região central (32x32 pixels) de mais de 220.000 imagens RGB de 96x96 pixels de lâminas histopatológicas em formato TIFF.

## ⚙️ Pré-processamento e Pipeline de Dados
- **Framework:** PyTorch.
- **Divisão de Dados:** Particionamento estratificado de 90% para treino e 10% para validação (Semente: 42).
- **Modos de Execução:** `proto` (amostragem de 5% dos dados para validação ágil de arquitetura) e `full` (conjunto de dados completo).
- **Data Augmentation e Transformações:** Recorte central (`CenterCrop` de 64x64 pixels na região de interesse), rotações aleatórias de até 90º, espelhamentos verticais e horizontais, e variações cromáticas (`ColorJitter` em fator 0.2). Normalização estatística através dos parâmetros padrão da ImageNet. Para o ViT, as imagens da região 36x36 foram redimensionadas para patches em 48x48.

## 🧠 Modelos Desenvolvidos e Avaliados
1. **Baseline CNN:** Rede neural convolucional simples com 4 blocos sequenciais (desenvolvida do zero com 32, 64, 128 e 256 filtros), usando ReLU, MaxPool e Dropout.
2. **ResNet-18 (Transfer Learning):** Rede residual ajustada para patologia através de pesos pré-treinados na ImageNet com fine-tuning integral (atualização de todos os ~11.1 milhões de parâmetros). **Este foi o modelo de melhor desempenho.**
3. **Visual Transformer (ViT):** Rede focada em mecanismo de atenção global profundo (12 camadas, 12 cabeças). Foi treinado utilizando otimizador AdamW, Cosine Annealing, e avaliado via Mapas de Saliência (Saliency Mapping) para maior interpretabilidade dos biomarcadores visuais.
4. **SimCLR (Aprendizado Auto-Supervisionado - SSL):** Extração de características visuais não supervisionadas via framework contrastivo. Demonstrado empíricamente como excelente regularizador em regimes de escassez de dados (*low-data regime* no modo proto).
5. **Ensemble FFT + ResNet-18:** Arquitetura de fusão tardia combinando uma extração de 78 atributos de frequência (FFT) e características espaciais da ResNet-18, ainda que empiricamente tenha mantido alta sobreposição de erros.

## 📊 Resultados Principais (Modo Full / Conjunto de Validação)

| Abordagem / Modelo | Acurácia (Val.) | ROC-AUC (Val.) |
| :--- | :--- | :--- |
| **ResNet-18** | 90,72% | 0,9709 |
| **BaselineCNN** | 89,05% | 0,9569 |
| **SimCLR + Baseline** | 89,02% | 0,9569 |
| **Visual Transformer (ViT)** | 84,65% | 0,9201 |

*Nota: O modelo ViT foi submetido no Kaggle e obteve a pontuação ROC-AUC de 0,9025 no leaderboard público.*

## 🚀 Trabalhos Futuros e Limitações
A principal limitação constatada pelo grupo se deu devido à barreira computacional de hardware para treinos longos em todas as mais de 220.000 imagens. Oportunidades propostas para evoluções futuras:
- Incorporação de rotinas de normalização colorimétrica de lâminas (método Macenko).
- Aumento de dados durante a etapa de inferência (*Test-Time Augmentation* - TTA).
- Utilização de modelos inicializados com pesos biomédicos específicos (*Pathology-ImageNet*).
- Submissão de ensembles heterogêneos liderados pela predição dos pesos ajustados da ResNet-18 integrados ao mecanismo de atenção global do ViT.

README.md
Exibindo README.md.
