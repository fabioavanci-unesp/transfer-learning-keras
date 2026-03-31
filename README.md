# Transfer Learning com Xception e Self-Attention 

## Visão Geral do Projeto

Este repositório contém um projeto de classificação de imagens para detecção de câncer de pele, utilizando técnicas avançadas de Deep Learning, incluindo Transfer Learning com a arquitetura Xception e a integração de mecanismos de Self-Attention. O objetivo é classificar sete tipos distintos de lesões cutâneas a partir do dataset HAM10000, contribuindo para o diagnóstico assistido por computador em dermatologia.

O trabalho é inspirado e fundamentado no artigo científico "Skin Cancer Detection Using Transfer Learning and Deep Attention Mechanisms" de Alotaibi e AlSaeed (2025) [1], que explora o impacto de mecanismos de atenção no desempenho de modelos de Transfer Learning para a detecção de câncer de pele.

## Estrutura do Repositório

*   `notebook.ipynb`: Notebook Jupyter contendo todo o código-fonte para pré-processamento de dados, construção do modelo, treinamento, avaliação e visualização de resultados.
*   `README.md`: Este arquivo, fornecendo uma visão geral do projeto.


## Funcionalidades e Metodologia

### 1. Pré-processamento de Imagens Avançado

O pipeline de pré-processamento de imagens e inclui:

*   **Center Crop e Redimensionamento:** Ajuste das imagens para um tamanho padrão (299x299 pixels) sem distorção.
*   **Filtragem Gaussiana:** Suavização de ruídos para realçar características importantes.
*   **Equalização de Histograma (YUV):** Melhoria do contraste para otimizar a visualização de detalhes.
*   **Remoção de Pelos (DullRazor):** Uma etapa crucial para eliminar artefatos que podem confundir o modelo, garantindo que o foco esteja nas características da lesão.
*   **Data Augmentation:** Aplicação de rotações e flips aleatórios para aumentar a diversidade do dataset de treinamento e melhorar a robustez do modelo.

### 2. Arquitetura do Modelo: Xception-SL (Self-Attention)

O coração do modelo é uma adaptação da rede convolucional **Xception**, pré-treinada no ImageNet, que é conhecida por sua eficiência e profundidade. A inovação reside na integração de uma camada customizada de **Self-Attention** (Xception-SL) após a extração de características pelo Xception. Esta camada permite que o modelo atribua maior peso a regiões da imagem que são mais relevantes para a classificação, emulando o processo de foco humano e melhorando a capacidade discriminativa do modelo, conforme sugerido por Alotaibi e AlSaeed [1].

### 3. Balanceamento de Dados

O dataset original HAM10000 é desbalanceado, com uma grande predominância de nevos melanocíticos (`nv`). Foi implementada uma estratégia de balanceamento que expande o dataset, resultando em uma distribuição uniforme de amostras para cada uma das 7 categorias de lesões.

## 4. Resultados e Discussão

O treinamento foi configurado para 50 épocas com técnicas de *Early Stopping* para evitar o *overfitting* e obteve uma acuracia de 74% e AUC 0.947. Tempo total de treinamento: 16 horas.
Desempenho geral pode ser considerado bom para um problema de 7 classes, a acurácia de 74% com AUC de 0.947 é um resultado expressivo. O AUC próximo de 1.0 indica que o modelo separa bem as classes no espaço de probabilidade, mesmo quando erra na classificação final. Do ponto de vista clínico, errar um mel é mais grave do que dar um falso alarme. Um recall de 0.58 no melanoma significa que 4 em cada 10 melanomas não são detectados. Para uma aplicação médica real, seria necessário ajustar o threshold dessa classe ou aplicar pesos de classe maiores no treinamento.

## Como Executar

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/fabioavanci-unesp/transfer-learning-keras.git
    cd transfer-learning-keras
    ```
2.  **Instale as Dependências:**
    Certifique-se de ter Python 3.x e `pip` instalados. As principais bibliotecas incluem `tensorflow`, `keras`, `numpy`, `pandas`, `matplotlib`, `opencv-python` e `scikit-learn`. Você pode instalá-las via `pip`:
    ```bash
    pip install -r requirements.txt # Se houver um arquivo requirements.txt
    # Ou instale manualmente:
    pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn kagglehub
    ```
3.  **Execute o Notebook:**
    Abra o `trabalho-1.ipynb` em um ambiente Jupyter (Jupyter Lab, Jupyter Notebook, VS Code com extensão Jupyter) e execute as células sequencialmente para replicar o pré-processamento, treinamento e avaliação do modelo.
    ```bash
    jupyter notebook trabalho-1.ipynb
    ```

## Referências

[1] ALOTAIBI, A.; ALSAEED, D. **Skin Cancer Detection Using Transfer Learning and Deep Attention Mechanisms**. Diagnostics (Basel), v. 15, n. 1, p. 99, jan. 2025. Disponível em: [https://pmc.ncbi.nlm.nih.gov/articles/PMC11720014/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11720014/). Acesso em: 30 mar. 2026.
