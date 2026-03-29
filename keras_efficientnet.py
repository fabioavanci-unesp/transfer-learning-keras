import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import kagglehub
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# FUNÇÕES DE PLOT E PRÉ-PROCESSAMENTO
# ==========================================
def plot_historico_completo(history1, history2):
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epoca_transicao = len(history1.history['accuracy'])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Treino', linewidth=2)
    plt.plot(epochs, val_acc, 'r--', label='Validação', linewidth=2)
    plt.axvline(x=epoca_transicao, color='k', linestyle=':', linewidth=2, label='Início Fine-Tuning (Fase 2)')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='-', alpha=0.3)

    # Gráfico de Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Treino', linewidth=2)
    plt.plot(epochs, val_loss, 'r--', label='Validação', linewidth=2)
    plt.axvline(x=epoca_transicao, color='k', linestyle=':', linewidth=2, label='Início Fine-Tuning (Fase 2)')
    plt.title('Evolução do Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.show()


def center_crop_and_resize(img, target_size=(380, 380)):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w // 2) - (min_dim // 2)
    start_y = (h // 2) - (min_dim // 2)
    img_cropped = img[start_y: start_y + min_dim, start_x: start_x + min_dim]
    return cv2.resize(img_cropped, target_size, interpolation=cv2.INTER_AREA)


import cv2
import matplotlib.pyplot as plt


def preprocess_ham10000_image(image_path, target_size=(380, 380), plot_comparison=False):
    # Carregar imagem
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")

    # Guardar a imagem original em RGB caso o usuário queira plotar
    if plot_comparison:
        img_original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Corte Central
    img_resized = center_crop_and_resize(img, target_size=target_size)

    # 2. Filtragem Gaussiana (suavização)
    img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0.5)

    # 3. Equalização de Histograma (via espaço YUV)
    img_yuv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # 4. Remoção de Pelos (Algoritmo DullRazor)
    gray_scale = cv2.cvtColor(img_equalized, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img_hair_removed = cv2.inpaint(img_equalized, hair_mask, 1, cv2.INPAINT_TELEA)

    # Converter de BGR (OpenCV) para RGB (Keras/Matplotlib)
    img_rgb = cv2.cvtColor(img_hair_removed, cv2.COLOR_BGR2RGB)

    # ==========================================
    # LÓGICA DE PLOTAGEM (Lado a Lado)
    # ==========================================
    if plot_comparison:
        plt.figure(figsize=(10, 5))

        # Plot da imagem original
        plt.subplot(1, 2, 1)
        plt.imshow(img_original_rgb)
        plt.title(f"Original\n({img_original_rgb.shape[1]}x{img_original_rgb.shape[0]})")
        plt.axis("off")

        # Plot da imagem pré-processada (cortada, sem pelos, equalizada)
        plt.subplot(1, 2, 2)
        plt.imshow(img_rgb)
        plt.title(f"Pré-processada\n({target_size[0]}x{target_size[1]})")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    # Normalizar pixels para [0, 1] para a rede neural
    img_normalized = img_rgb / 255.0

    return img_normalized

# ==========================================
# 1. DOWNLOAD E CARREGAMENTO DOS DADOS
# ==========================================
print("Iniciando download do dataset HAM10000...")
# Usando a forma padrão atual do kagglehub para carregar como pandas
df_meta = kagglehub.load_dataset(
    kagglehub.KaggleDatasetAdapter.PANDAS,
    "kmader/skin-cancer-mnist-ham10000",
    "HAM10000_metadata.csv",
)

dataset_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
image_paths = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
image_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in image_paths}

df_meta['path'] = df_meta['image_id'].map(image_dict)

classes = df_meta['dx'].unique()
class_dict = {classes[i]: i for i in range(len(classes))}
df_meta['label'] = df_meta['dx'].map(class_dict)

print(f"Dataset carregado. Total de imagens: {len(df_meta)}")
num_classes = len(classes)

# ==========================================
# 2. PREPARAÇÃO DOS DADOS (Train/Test Split de Caminhos)
# ==========================================
print("Dividindo os dados de treino e validação...")

# Pegamos apenas os caminhos e as labels em formato de número inteiro
X_paths = df_meta['path'].values
y_labels = df_meta['label'].values

# Dividimos os caminhos e as labels (e não as imagens em si)
X_train_paths, X_val_paths, y_train_int, y_val_int = train_test_split(
    X_paths, y_labels, test_size=0.20, random_state=42, stratify=y_labels
)

# Convertendo para One-Hot Encoding
y_train = to_categorical(y_train_int, num_classes=num_classes)
y_val = to_categorical(y_val_int, num_classes=num_classes)

print(f"Total de imagens de treino: {len(X_train_paths)}")
print(f"Total de imagens de validação: {len(X_val_paths)}")


# ==========================================
# 2.1 CRIANDO O DATA GENERATOR
# ==========================================
class HAM10000Generator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=16, target_size=(380, 380)):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        # Retorna o número de batches por época
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Pega os caminhos e labels do lote (batch) atual
        batch_paths = self.image_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_x = []
        for path in batch_paths:
            # Usa sua função original de pré-processamento (certifique-se de que ela está definida acima)
            img = preprocess_ham10000_image(path, target_size=self.target_size, plot_comparison=False)
            batch_x.append(img)

        return np.array(batch_x), np.array(batch_labels)


# Instanciando os geradores
batch_size = 16
train_generator = HAM10000Generator(X_train_paths, y_train, batch_size=batch_size)
val_generator = HAM10000Generator(X_val_paths, y_val, batch_size=batch_size)

# ==========================================
# 3. MODELAGEM E TREINAMENTO (FASE 1)
# ==========================================

# 1. CALCULAR PESOS DAS CLASSES (Para lidar com o desbalanceamento)
# Isso calcula o peso inversamente proporcional à frequência da classe
# Classes raras terão pesos altos (ex: 5.0), a classe 'nv' terá peso baixo (ex: 0.2)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weights_dict = dict(enumerate(class_weights_array))
print("Pesos atribuídos a cada classe:", class_weights_dict)

# 2. CRIAR CAMADAS DE DATA AUGMENTATION NATIVAS
# Essas operações ocorrem na GPU e são muito rápidas
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"), # Espelha horizontal e verticalmente
    RandomRotation(0.2),                   # Rotaciona até 20% (aprox 72 graus)
    RandomZoom(0.1),                       # Aplica zoom in/out de até 10%
], name='data_augmentation')

print("Construindo o modelo EfficientNet-B4...")

base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(380, 380, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(380, 380, 3))

# Passar o input pela camada de augmentation antes de ir para a rede base
x = data_augmentation(inputs)

# Agora a rede recebe as imagens "aumentadas" e rotacionadas
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Descobre quantos núcleos de CPU a máquina virtual do Colab/Kaggle te alocou
num_cores = os.cpu_count()
print(f"Núcleos de CPU disponíveis para os workers: {num_cores}")

# 4. TREINAR USANDO O GERADOR
history_phase1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    workers=num_cores,               # Usa múltiplas threads para carregar as imagens mais rápido
    use_multiprocessing=False # No Windows, deixe False para evitar travamentos com o OpenCV
)

# ==========================================
# 4. FINE-TUNING PROFUNDO (FASE 2)
# ==========================================
print("\n--- INICIANDO FASE 2 ---")
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history_phase2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks,
    workers=num_cores,
    use_multiprocessing=False
)

# ==========================================
# 5. SALVAR E PLOTAR
# ==========================================
model.save('modelo_ham10000_efficientnetb4.h5')
print("Treinamento finalizado e modelo salvo com sucesso!")

plot_historico_completo(history_phase1, history_phase2)