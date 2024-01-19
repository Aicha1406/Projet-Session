import os
import sys
import time

import mplt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# en activant ici cest le webcame de l'ordinateur que vous utiliser
cap = cv2.VideoCapture(0)
# ici cest la video
#cap = cv2.VideoCapture("test3.avi")

kernel_blur = 5
seuil = 15
surface = 5000

# le taux d'apprentissage (learning rate)
learning_rate = 0.1

ret, originale = cap.read()
originale = cv2.cvtColor(originale, cv2.COLOR_BGR2GRAY)
originale = cv2.GaussianBlur(originale, (kernel_blur, kernel_blur), 0)
kernel_dilate = np.ones((5, 5), np.uint8)

precisions = []  # Liste pour stocker les précisions à chaque itération
vraies_classes = []  # Liste pour stocker les classes réelles
classes_predites = []  # Liste pour stocker les classes prédites

# pour les poids du modèle
modele_poids = 0.5



while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (kernel_blur, kernel_blur), 0)

    # ici je fait la différence entre l'image récupérée et celle avec le flou
    mask = cv2.absdiff(originale, gray)
    # pour détecter les objets

    # ici c'est augmenter l'intensité du flou
    mask = cv2.threshold(mask, seuil, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("frame", mask)
    mask = cv2.dilate(mask, kernel_dilate, iterations=3)
    contours, nada = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_contour = frame.copy()
    # cette fonction permet de dessiner mes contours
    for c in contours:
        cv2.drawContours(frame_contour, [c], 0, (0, 255, 0), 5)
        # ici je calcule le contour et sélectionne la surface
        if cv2.contourArea(c) < surface:
            continue
            # pour entourer le mouvement avec le rectangle
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Calcul de la précision
    total_contours = len(contours)
    gros_contours = sum(cv2.contourArea(c) >= surface for c in contours)
    precision_calcul = gros_contours / total_contours if total_contours > 0 else 0

    precisions.append(precision_calcul)  # Ajouter la précision à la liste
    vraies_classes.append(1 if precision_calcul >= 0.5 else 0)
    classes_predites.append(1 if precision_calcul >= 0.5 else 0)

    #  d'utilisation du modèle
    prediction_modele = modele_poids * precision_calcul

    # Mise à jour du seuil avec le taux d'apprentissage
    seuil = seuil - learning_rate * (precision_calcul - 0.5)

    # Afficher le taux d'apprentissage
    print("Taux d'apprentissage:", learning_rate)

    # Afficher la matrice de confusion chaque n itérations (par exemple, n=10)
    if len(precisions) % 10 == 0:
        matrice_confusion = confusion_matrix(vraies_classes, classes_predites)
        print("Matrice de confusion:")
        print(matrice_confusion)

        # Afficher la matrice de confusion avec seaborn
        sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Classe Prédite')
        plt.ylabel('Classe Réelle')
        plt.title('Matrice de Confusion')
        plt.show()

    # Tracer le graphe de précision
    plt.plot(precisions)
    plt.xlabel('Itération')
    plt.ylabel('Précision')
    plt.title('Graphe de précision')
    mplt.show()

    # Tracer le graphe du taux d'apprentissage
    plt.plot('taux_apprentissage_liste')
    plt.xlabel('Itération')
    plt.ylabel('Taux d\'Apprentissage')
    plt.title('Graphe du Taux d\'Apprentissage')
    mplt.show()


    originale = gray

    cv2.putText(frame, "[o|l]seuil: {:.2f}  [p|m]blur: {:.2f}  [i|k]surface: {:.2f}".format(seuil, kernel_blur, surface),
                (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Précision: {:.2f}".format(precision_calcul), (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                (0, 255, 255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("contour", frame_contour)
    cv2.imshow("mask", mask)
    intrus = 0
    key = cv2.waitKey(30) & 0xFF
    # permet de diminuer ou augmenter la taille du flou
    if key == ord('q'):
        break
    if key == ord('p'):
        kernel_blur = min(43, kernel_blur + 2)
    if key == ord('m'):
        kernel_blur = max(1, kernel_blur - 2)
    # if key == ord('i'):
    #   surface += 1000
    # if key == ord('k'):
    surface = max(1000, surface - 1000)

cap.release()
cv2.destroyAllWindows()
