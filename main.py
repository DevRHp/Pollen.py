# made by:  Rafael Prudencio
# email:    devprudencio@gmail.com
# linkedin: https://www.linkedin.com/in/rafael-prudencio-4222382b6/
# github:   https://github.com
# version:  1.7.4
# date:     11/09/2024 - 02/06/2025

# ========================================================================= #

import cv2
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog
import shutil

print("TensorFlow version:", tf.__version__)

# Definir o diretório do projeto
base_path = Path("C:/Users/Administrador/Desktop/Pollen.py")

# ========================================================================= #

# Definir pastas de classes
class_dirs = [
    base_path / "Acacia_img",
    base_path / "Cerastium_img",
    base_path / "Cirsium_img",
    base_path / "Clematis_img" # <-- Adicione aqui as pastas das demais classes
]

# ========================================================================= #

# Ajusta a resolução das imagens
width = 128
height = 128

# Define o limiar para binarização
threshold = 0.5

# Define o ângulo máximo de rotação
max_rotation_angle = 15

# ========================================================================= #

# Função para rotacionar imagens
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Função para carregar e pré-processar imagens
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Aplica rotação aleatória
            angle = np.random.uniform(-max_rotation_angle, max_rotation_angle)
            img = rotate_image(img, angle)
            img = cv2.resize(img, (width, height))
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold(img_gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
            img_bin = np.expand_dims(img_bin, axis=-1)  # Adiciona dimensão do canal
            img_hsv = img_hsv / 255.0  # Normaliza a imagem HSV
            img_bin = img_bin / 255.0  # Normaliza a imagem binarizada
            combined_img = np.concatenate((img_bin, img_hsv), axis=-1)  # Combina as imagens
            images.append(combined_img)
            labels.append(label)
    return images, labels

# Carregar imagens de cada classe
X = []
y = []

for i, class_dir in enumerate(class_dirs):
    label = [0] * len(class_dirs)
    label[i] = 1
    images, labels = load_images_from_folder(class_dir, label)
    X.extend(images)
    y.extend(labels)

X = np.array(X)
y = np.array(y)

# ========================================================================= #

# Estrutura da rede neural convolucional
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(height, width, 4)),  # 1 canal (binarizado) + 3 canais (HSV)
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_dirs), activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# ========================================================================= #

# Treinar o modelo
EPOCHS = 150
BS = 32

if len(X) == 0:
    print("Nenhuma imagem capturada para treinamento!")
else:
    model.fit(X, y, 
              epochs=EPOCHS,
              batch_size=BS,
              validation_split=0.2)
    print("Treinamento concluído!")

    # ========================================================================= #

    # Salvar o modelo treinado
    model_save_path = base_path / "modelo_treinado.keras"
    model.save(model_save_path)
    print(f"Modelo salvo em: {model_save_path}")

# ========================================================================= #

# Função para carregar uma imagem para predição
def load_image_for_prediction(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (width, height))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(img_gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
        img_bin = np.expand_dims(img_bin, axis=-1)  # Adiciona dimensão do canal
        img_hsv = img_hsv / 255.0  # Normaliza a imagem HSV
        img_bin = img_bin / 255.0  # Normaliza a imagem binarizada
        combined_img = np.concatenate((img_bin, img_hsv), axis=-1)  # Combina as imagens
        return combined_img
    else:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None

# Função para exibir a imagem e a predição na interface gráfica
def display_prediction(image_path, prediction, confidence):
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)

    label_img.configure(image=img_tk)
    label_img.image = img_tk
    label_prediction.configure(text=f"Predição: {prediction} - Confiança: {confidence:.2f}")

# Função para escolher uma imagem e fazer a predição
def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        print(f"Carregando imagem de validação: {file_path}")  # Log de depuração
        validation_image = load_image_for_prediction(file_path)
        if validation_image is not None:
            validation_image = np.expand_dims(validation_image, axis=0)  # Adiciona dimensão do batch

            # Fazer predição
            pred = model.predict(validation_image)
            labels = ["Acacia", "Cerastium", "Cirsium", "Clematis"]  # Adicione mais labels aqui conforme o número de classes
            predicted_label = labels[np.argmax(pred)]
            confidence = np.max(pred)

            display_prediction(file_path, predicted_label, confidence)

            # Copiar a imagem para a pasta correspondente à predição
            target_dir = base_path / f"{predicted_label}_img"
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, target_dir / Path(file_path).name)
            print(f"Imagem copiada para: {target_dir}")
        else:
            print(f"Não foi possível carregar a imagem: {file_path}")

# Configuração da interface gráfica
root = ctk.CTk()
root.title("Predição de Imagens")
root.geometry("400x500")

frame = ctk.CTkFrame(root)
frame.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")

label_img = ctk.CTkLabel(frame)
label_img.grid(row=0, column=0, padx=20, pady=20)

label_prediction = ctk.CTkLabel(root, text="Predição: ")
label_prediction.grid(row=1, column=0, pady=10)

button_choose = ctk.CTkButton(root, text="Escolher Imagem", command=choose_image)
button_choose.grid(row=2, column=0, pady=20)

# Centralizar a imagem e os widgets
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.grid_rowconfigure(0, weight=1)

# Definir caminho da pasta de validação
validation_base_path = base_path / "data_val_pollen"

root.mainloop()
