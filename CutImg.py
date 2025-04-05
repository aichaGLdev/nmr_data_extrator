from flask import Flask, request, jsonify # send_file # Importation des modules Flask nécessaires
import cv2  # Importation de la bibliothèque OpenCV
import os  # Importation du module os pour la manipulation de fichiers et de répertoires
import numpy as np  # Importation de la bibliothèque NumPy pour le traitement des tableaux
import random

def convert_to_bin_image(image, MaxValue, BinaryType):
    #convertir l'image de l'espace de couleur RGB a l'espace de couleur en niveaux de gris (grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray contiendra une version en niveaux de gris de l'image image, où chaque pixel aura une valeur représentant son niveau de luminance, allant du noir (0) au blanc (255).
    #BinaryType cv2.THRESH_BINARY pixel blanc ou cv2.THRESH_BINARY_INV (pixel noir
    image_bin = cv2.adaptiveThreshold(gray, MaxValue, cv2.ADAPTIVE_THRESH_MEAN_C, BinaryType, 7, 10)
    # image_bin, où les pixels ont des valeurs binaires (0 ou MaxValue)
    return image_bin
def icvpr_cca_by_two_pass(binImg ,labelImg ):
    label = 1
    labelSet = [0, 1]
    maxLabel = 0

    # Définition des dimensions de l'image
    rows = len(binImg) - 1
    cols = len(binImg[0]) - 1

    # Première passe
    for i in range(1, rows):
        for j in range(1, cols):
            if binImg[i][j] == 1:
                neighborLabels = []
                leftPixel = labelImg[i][j - 1]
                upPixel = labelImg[i - 1][j]
                leftupPixel = labelImg[i - 1][j - 1]
                rightupPixel = labelImg[i - 1][j + 1]

                if leftPixel > 1:
                    neighborLabels.append(leftPixel)
                if upPixel > 1:
                    neighborLabels.append(upPixel)
                if leftupPixel > 1:
                    neighborLabels.append(leftupPixel)
                if rightupPixel > 1:
                    neighborLabels.append(rightupPixel)

                if not neighborLabels:
                    labelSet.append(label)
                    labelImg[i][j] = label
                    labelSet[label] = label
                    label += 1
                else:
                    neighborLabels.sort()
                    smallestLabel = neighborLabels[0]
                    labelImg[i][j] = smallestLabel

                    for k in range(1, len(neighborLabels)):
                        tempLabel = neighborLabels[k]
                        oldSmallestLabel = labelSet[tempLabel]
                        if oldSmallestLabel > smallestLabel:
                            labelSet[oldSmallestLabel] = smallestLabel
                            oldSmallestLabel = smallestLabel
                        elif oldSmallestLabel < smallestLabel:
                            labelSet[smallestLabel] = oldSmallestLabel

    # Mise à jour des labels équivalents
    for i in range(2, len(labelSet)):
        curLabel = labelSet[i]
        preLabel = labelSet[curLabel]
        while preLabel != curLabel:
            curLabel = preLabel
            preLabel = labelSet[preLabel]
        labelSet[i] = curLabel

    # Deuxième passe
    for i in range(rows):
        for j in range(cols):
            pixelLabel = labelImg[i][j]
            labelImg[i][j] = labelSet[pixelLabel]
            if pixelLabel > maxLabel:
                maxLabel = pixelLabel

    return maxLabel
def icvpr_get_random_color():
    r = int(255 * (random.random() / (1.0 + random.random())))
    g = int(255 * (random.random() / (1.0 + random.random())))
    b = int(255 * (random.random() / (1.0 + random.random())))
    return (b, g, r)

def icvpr_label_color(label_img):
    if label_img is None or label_img.dtype != np.int32:
        return np.zeros_like(label_img, dtype=np.uint8)  # Renvoyer une image vide avec la même forme que l'entrée

    colors = {}

    rows, cols = label_img.shape

    color_label_img = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            pixel_value = label_img[i, j]
            if pixel_value > 1:
                if pixel_value not in colors:
                    colors[pixel_value] = icvpr_get_random_color()
                color = colors[pixel_value]
                color_label_img[i, j] = color

    return color_label_img




def cmp_x(a, b):
    return a[0] < b[0]

def cmp_y(a, b):
    return a[1] < b[1]

def point_cmp_row(a, b):
    return a[0].y < b[0].y

def point_cmp_col(a, b):
    return a[0].x < b[0].x



def cut_num(image, save_filename):
    # Convertir l'image en niveaux de gris et en image binaire
    grayImage = convert_to_bin_image(image, 255, cv2.THRESH_BINARY)
    binImage = convert_to_bin_image(image, 1, cv2.THRESH_BINARY_INV)

    # Définir les coordonnées pour l'échelle
    scale_row_start, scale_row_end, scale_col_start, scale_col_end = None, None, None, None

    # Étiquetage des composantes connectées
    labelImg = icvpr_cca_by_two_pass(binImage)
    max_label = np.max(labelImg)
    # Initialisation de la liste des étiquettes
    labels = [[] for _ in range(max_label + 1)]

    # Remplissage de la liste des étiquettes
    for i in range(labelImg.shape[0]):
        for j in range(labelImg.shape[1]):
            labels[labelImg[i, j]].append((i, j))

    # Conversion de l'image d'étiquettes en image de couleurs
    labelImg *= 10
    colorLabelImg = icvpr_label_color(labelImg)

    # Parcourir les étiquettes
    for i in range(2, len(labels)):
        if len(labels[i]) < 5:
            # Supprimer les petites étiquettes
            for x, y in labels[i]:
                colorLabelImg[x, y] = (0, 0, 0)
            continue

        elif len(labels[i]) < 500:
            continue

        # Trouver les coins des rectangles
        sort_by_row = sorted(labels[i], key=lambda x: x[0])
        sort_by_col = sorted(labels[i], key=lambda x: x[1])
        x1, y1 = sort_by_row[0]
        x2, y2 = sort_by_row[-1]
        if 10 <= x2 - x1 <= 30 and 1600 <= y2 - y1:
            print("Marquer l'échelle.")
            scale_row_start, scale_row_end, scale_col_start, scale_col_end = x1, x2, y1, y2

    # Enregistrement des chiffres
    NumSet = []
    for i in range(len(labels)):
        if 7 <= len(labels[i]) <= 300:
            sort_by_row = sorted(labels[i], key=lambda x: x[0])
            sort_by_col = sorted(labels[i], key=lambda x: x[1])
            x1, y1 = sort_by_row[0]
            x2, y2 = sort_by_row[-1]
            if scale_row_end < x1 < scale_row_end + 70:
                NumSet.append((y1, x1, y2, x2))

    # Tri des chiffres par colonne
    NumSet.sort(key=lambda x: x[0])

    for idx, (y1, x1, y2, x2) in enumerate(NumSet):
        # Récupérer le chiffre
        temp = grayImage[x1:x2 + 1, y1:y2 + 1]
        # Enregistrer le chiffre
        filename = f"{save_filename}_{idx}.bmp"
        cv2.imwrite(filename, temp)


def scale_detect(image, ocr):
    gray_image = convert_to_bin_image(image, 255, cv2.THRESH_BINARY)
    bin_image = convert_to_bin_image(image, 1, cv2.THRESH_BINARY_INV)
    scale_row_start, scale_row_end, scale_col_start, scale_col_end = 0, 0, 0, 0

    # Connected component labeling
    label_img = cv2.connectedComponents(bin_image)[1]
    max_label = icvpr_cca_by_two_pass(bin_image ,label_img)

    labels = [[] for _ in range(max_label + 1)]
    for i in range(label_img.shape[0]):
        for j in range(label_img.shape[1]):
            labels[label_img[i, j]].append((i, j))

    # Show result
    label_img = label_img * 10
    gray_img = label_img.astype(np.uint8)

    color_label_img = label_img.copy()
    color_label_img = icvpr_label_color(label_img)

    for i in range(2, len(labels)):
        if len(labels[i]) < 5:
            for k in range(len(labels[i])):
                color_label_img[labels[i][k][0], labels[i][k][1]] = (0, 0, 0)
            continue

        elif len(labels[i]) < 500:
            continue

        sort_by_row = sorted(labels[i], key=lambda x: x[0])
        sort_by_col = sorted(labels[i], key=lambda x: x[1])
        x1, y1 = sort_by_row[0]
        x2, y2 = sort_by_row[-1]

        if x2 - x1 <= 30 and y2 - y1 >= 1600:
            print("Mark out the scale.")
            scale_row_start = x1
            scale_row_end = x2
            scale_col_start = y1
            scale_col_end = y2
    # Compute scale column positions
    scale_col_pos = []
    scan_line = scale_row_start + (scale_row_end - scale_row_start) * 2 // 3
    print("scan_line", scan_line)
    #print("LEN color_label_img", color_label_img.shape)
    for i in range(scale_col_start, scale_col_end + 1):
        if any(color_label_img[scan_line, i] > 0):
            cnt = 0
            for k in range(i, scale_col_end + 1):
                if any(color_label_img[scan_line, k] > 0):
                    cnt += 1
                else:
                    new_pos = i + cnt // 2
                    if not scale_col_pos or new_pos != scale_col_pos[-1]:  # Vérifier la redondance
                        scale_col_pos.append(new_pos)
                    i += cnt
                    break
            i += 3

    unit_pixels = (scale_col_pos[-1] - scale_col_pos[0]) // (len(scale_col_pos) - 1)
    # Record number positions
    num_set = []
    max_width, max_height = 0, 0  # Step 1

    for i in range(len(labels)):
        if len(labels[i]) > 300 or len(labels[i]) < 3:
            continue

        sort_by_row = sorted(labels[i], key=lambda x: x[0])
        sort_by_col = sorted(labels[i], key=lambda x: x[1])
        x1 = sort_by_row[0][0]
        y1 = sort_by_col[0][1]
        x2 = sort_by_row[-1][0]
        y2 = sort_by_col[-1][1]

        if x1 > scale_row_end and x1 < scale_row_end + 50:
            num_set.append(((y1, x1), (y2, x2)))
            color_label_img = cv2.rectangle(color_label_img, (y1, x1), (y2, x2), (0, 0, 255), 1)

            # Step 1: Find maximum width and height
            width = y2 - y1 + 1
            height = x2 - x1 + 1
            max_width = max(max_width, width)
            max_height = max(max_height, height)


    num1 = [gray_image[num[0][1]:num[1][1], num[0][0]:num[1][0]] for num in num_set[1:3]]
    num1_structured = [np.repeat(num[:, :, np.newaxis], 3, axis=2) for num in
                       num1]  # Structurer les images en (12, 18, 3)

    num2 = [gray_image[num[0][1]:num[1][1] + 1, num[0][0]:num[1][0] + 1] for num in num_set[3:5]]
    num2_structured = [np.repeat(num[:, :, np.newaxis], 3, axis=2) for num in
                       num2]  # Structurer les images en (12, 18, 3)

    num_first = ocr.DetectNum(num1_structured)
    num_second = ocr.DetectNum(num2_structured)
    num_first = float(num_first)
    num_second = float(num_second)

    col_start = scale_col_pos[1]

    col_start_num = float(num_first)
    col_end_num = float(num_second)

    unit = col_end_num - col_start_num

    return col_start, col_start_num, unit_pixels, unit

