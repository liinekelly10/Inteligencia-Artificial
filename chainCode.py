import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import normalizeImage, verifyNeighborhood
import pandas as pd

# ---- CONFIGURAÇÕES ----
dim = (300,300)
path = './numbers/number1/'   # pasta das imagens
L = 256                       # comprimento alvo para normalização

# ---- FUNÇÃO DE NORMALIZAÇÃO ----
def normalize_chaincode(seq, L):
    """Ajusta a sequência para ter comprimento L"""
    n = len(seq)
    if n == 0:
        return [0] * L
    if n > L:
        # reduz proporcionalmente (interpolação simples)
        idx = np.linspace(0, n-1, L, dtype=int)
        return [seq[i] for i in idx]
    else:
        # repete valores até atingir L
        reps = np.linspace(0, n-1, L, dtype=int)
        return [seq[i] for i in reps]

# ---- EXTRAÇÃO E NORMALIZAÇÃO ----
all_sequences = []
original_sequences = []

for r, d, f in os.walk(path):
    for filename in f:

        image = cv2.imread(os.path.join(path, filename))
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        imageBin = 255 - image[:, :, 0]
        newIm = normalizeImage((imageBin > 100) * 1)
        imCopy = np.copy(newIm)

        imPlot = np.zeros(np.shape(image))
        imPlot[:, :, 0] = imPlot[:, :, 1] = imPlot[:, :, 2] = imCopy

        kernel = np.ones((3, 3), np.uint8)
        newIm = cv2.dilate(newIm, kernel, iterations=1) - newIm
        newIm = cv2.resize(newIm, dim, interpolation=cv2.INTER_AREA)

        max_xy = np.where(newIm == 255)
        startPoint = (max_xy[0][0], max_xy[1][0])

        ChainCode = []
        SignalLenght = []
        counter = 0

        point = verifyNeighborhood(newIm, startPoint, 4,
                                   counter=counter,
                                   ChainCode=ChainCode,
                                   SignalLenght=SignalLenght)

        while (point != startPoint):
            # desenha o ponto atual no contorno
            cv2.circle(imPlot, (point[1], point[0]), int(3), (0, 0, 255), 4)
            cv2.imshow('image', imPlot)
            cv2.waitKey(10)  # tempo de espera (10ms) para ver o movimento

            # continua verificando os próximos pontos
            point = verifyNeighborhood(newIm, point, 4,
                                       counter=counter,
                                       ChainCode=ChainCode,
                                       SignalLenght=SignalLenght)

        # salva sequência original
        original_sequences.append(ChainCode)

        # normaliza
        norm_seq = normalize_chaincode(ChainCode, L)
        all_sequences.append(norm_seq)

# ---- VISUALIZAÇÃO ----
for i in range(len(original_sequences)):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.plot(original_sequences[i])
    plt.subplot(1,2,2)
    plt.title("Normalizada (L=256)")
    plt.plot(all_sequences[i])
    plt.show()

# ---- SALVAR EM CSV ----
df = pd.DataFrame(all_sequences)
df.to_csv("dataset_chaincodes.csv", index=False)
print("Dataset salvo como dataset_chaincodes.csv")
