from CutImg import convert_to_bin_image, icvpr_cca_by_two_pass, icvpr_label_color
import cv2
import numpy as np

def detect_horizontal_bracket_andNum(image, ocr, numbers, bracket_pos):
    grayImage = convert_to_bin_image(image, 255, cv2.THRESH_BINARY)
    binImage = convert_to_bin_image(image, 1, cv2.THRESH_BINARY_INV)

    #étiqueter les pixels de l'image en fonction des labels obtenus par l'algorithme de composantes connectée
    labelImg = cv2.connectedComponents(binImage)[1]
    maxLabel = icvpr_cca_by_two_pass(binImage, labelImg)
    #organiser et regrouper les pixels de l'image étiquetée en fonction des labels obtenus par l'algorithme de composantes connectée
    labels = [[] for _ in range(maxLabel + 1)] #crée une liste contenant maxLabel + 1 sous-listes vides
    for i in range(labelImg.shape[0]):
        for j in range(labelImg.shape[1]):
            labels[labelImg[i, j]].append((i, j))

    # show result
    labelImg = labelImg * 10
    grayImg = labelImg.astype(np.uint8)
    cv2.imwrite("result.bmp", grayImg)

    colorLabelImg = icvpr_label_color(labelImg)

    NumSet = []
    # filtrer les composantes connectées identifiées dans l'image
    for i in range(2, len(labels)):
        if len(labels[i]) > 700 or len(labels[i]) < 5:
            labels[i].clear()
            continue

        sort_by_row = sorted(labels[i], key=lambda x: x[0])
        sort_by_col = sorted(labels[i], key=lambda x: x[1])
        x1 = sort_by_row[0][0]
        y1 = sort_by_col[0][1]
        x2 = sort_by_row[-1][0]
        y2 = sort_by_col[-1][1]
        if 1280 <= x1 <= 1430 and 5 <= x2 - x1 and 12 <= y2 - y1 <= 200:
            NumSet.append(((y1, x1), (y2, x2)))

    NumSet.sort(key=lambda x: x[0][1], reverse=True)
    horizontalbase = NumSet[-1][0][1]
    brackets = []
    bracketnumbers = []
    Singlenum = []
    lenk = []
    wp = 0
    for i in range(len(NumSet)):
        if horizontalbase - 5 <= NumSet[i][0][1] <= horizontalbase + 5:
            brackets.append(NumSet[i])
        else:
            bracketnumbers.append(NumSet[i])

    brackets.sort(key=lambda x: x[0][0])
    bracketnumbers.sort(key=lambda x: x[0][0])
    for i in range(len(brackets)):
        bracket_base = brackets[i][0][1] + 3
        basecolor = colorLabelImg[brackets[i][1][1], (brackets[i][0][0] + brackets[i][1][0]) // 2]
        for k in range(brackets[i][0][1], brackets[i][1][1] + 1):
            if tuple(basecolor) != (0, 0, 0):
                break
            basecolor = colorLabelImg[k, (brackets[i][0][0] + brackets[i][1][0]) // 2]
        temp_brackets = []
        c = brackets[i][0][0]
        while c <= brackets[i][1][0]:
            if tuple(colorLabelImg[bracket_base, c]) == tuple(basecolor):
                temp_brackets.append(c)
                while tuple(colorLabelImg[bracket_base, c]) == tuple(basecolor):
                    c += 1
            else:
                c += 1
        for k in range(len(temp_brackets) - 1):
            bracket_pos.append((temp_brackets[k], temp_brackets[k + 1]))
            if temp_brackets[k + 1] - temp_brackets[k] <= 4:
                continue

            if bracket_pos and (temp_brackets[k], temp_brackets[k + 1]) == bracket_pos[-1]:
                continue
    for i in range(1, len(bracketnumbers)):
        if abs(bracketnumbers[i - 1][0][0] - bracketnumbers[i][0][0]) < 5:
            wp += 1
        else:
            lenk.append(wp)
            wp = 0

    lenk.append(wp)

    count1 = 0
    for i in range(len(bracketnumbers)):
        Singlenum.append(bracketnumbers[i])
        count2 = lenk[count1]
        if len(Singlenum) == count2 + 1:
            Singlenum.sort(key=lambda x: x[0][1], reverse=True)
            num_mat = []
            for k in range(count2 + 1):
                temp = grayImage[Singlenum[k][0][1]:Singlenum[k][1][1] + 1, Singlenum[k][0][0]:Singlenum[k][1][0] + 1]
                temp = cv2.transpose(temp)
                temp = cv2.flip(temp, 1)
                cv2.imwrite("temp.png", temp)
                temp = cv2.imread("temp.png")
                num_mat.append(temp)


            detectnum = ocr.DetectNumero(num_mat)
            # Convertir chaque tableau NumPy en chaîne de caractères
            detectnum_str = [str(num[0][0]) for num in detectnum]

            # Concaténer les éléments de detectnum_str en une seule chaîne de caractères
            detectnum_concatenated = ''.join(detectnum_str)

            # Calculer iPos à partir de la longueur de detectnum_concatenated
            iPos = len(detectnum_concatenated) - 2

            # Concaténer les parties de detectnum_concatenated et ajouter un point décimal
            numstr = detectnum_concatenated[:iPos] + "." + detectnum_concatenated[iPos:]

            # Afficher la chaîne de caractères numstr
            print(numstr)
            numbers.append(numstr)

            Singlenum.clear()
            count2 = 0
            count1 += 1

    cv2.imwrite("color.bmp", colorLabelImg)

def detectcntrnum(image, ocr, numbers, bracket_pos):
    grayImage = convert_to_bin_image(image, 255, cv2.THRESH_BINARY)
    binImage = convert_to_bin_image(image, 1, cv2.THRESH_BINARY_INV)

    # connected component labeling
    labelImg = cv2.connectedComponents(binImage)[1]
    maxLabel = icvpr_cca_by_two_pass(binImage, labelImg)
    labels = [[] for _ in range(maxLabel + 1)]
    for i in range(labelImg.shape[0]):
        for j in range(labelImg.shape[1]):
            labels[labelImg[i, j]].append((i, j))

    # show result
    labelImg = labelImg * 10
    grayImg = labelImg.astype(np.uint8)
    cv2.imwrite("result.bmp", grayImg)

    colorLabelImg = icvpr_label_color(labelImg)

    NumSet = []

    for i in range(2, len(labels)):
        if len(labels[i]) > 400 or len(labels[i]) < 5:
            labels[i].clear()
            continue

        sort_by_row = sorted(labels[i], key=lambda x: x[0])
        sort_by_col = sorted(labels[i], key=lambda x: x[1])
        x1 = sort_by_row[0][0]
        y1 = sort_by_col[0][1]
        x2 = sort_by_row[-1][0]
        y2 = sort_by_col[-1][1]

        if 20 <= x1 <= 125 and 5 <= x2 - x1 and 10 <= y2 - y1 <= 150:
            cv2.rectangle(colorLabelImg, (y1, x1), (y2, x2), (0, 0, 255))
            NumSet.append(((y1, x1), (y2, x2)))


    NumSet.sort(key=lambda x: x[0][0])
    horizontalbase = NumSet[-1][0][1]

    brackets = []
    bracketnumbers = []
    Singlenum = []  # Initialisation de la variable Singlenum
    lenk = []  # Initialisation de la variable lenk
    count1 = 0  # Initialisation de la variable count1
    for i in range(len(NumSet)):
        bracketnumbers.append(NumSet[i])


    for i in range(len(bracketnumbers)):
        Singlenum.append(bracketnumbers[i])
        if len(Singlenum) == 3:
            Singlenum.sort(key=lambda x: x[0][0], reverse=True)
            Singlenum.sort(key=lambda x: x[0][1], reverse=True)
            num_mat = []
            for k in range(3):
                temp = grayImage[Singlenum[k][0][1]:Singlenum[k][1][1] + 1, Singlenum[k][0][0]:Singlenum[k][1][0] + 1]
                temp = cv2.transpose(temp)
                temp = cv2.flip(temp, 1)
                cv2.imwrite("temp.png", temp)
                temp = cv2.imread("temp.png")
                num_mat.append(temp)

            detectnum = ocr.DetectNumero(num_mat)
            iPos = len(detectnum) - 2
            numstr = detectnum[:iPos] + "." + detectnum[iPos:]
            numbers.append(numstr)

            Singlenum.clear()
            count2 = 0
            count1 += 1



def detect_wntr_num(image, ocr, numbers, bracket_pos):
    grayImage = convert_to_bin_image(image, 255, cv2.THRESH_BINARY)
    binImage = convert_to_bin_image(image, 1, cv2.THRESH_BINARY_INV)

    labelImg = cv2.connectedComponents(binImage)[1]
    maxLabel = icvpr_cca_by_two_pass(binImage, labelImg)

    labels = [[] for _ in range(maxLabel + 1)]
    for i in range(labelImg.shape[0]):
        for j in range(labelImg.shape[1]):
            labels[labelImg[i, j]].append((i, j))

    labelImg = labelImg * 10
    grayImg = labelImg.astype(np.uint8)
    cv2.imwrite("result.bmp", grayImg)

    colorLabelImg = labelImg.copy()
    colorLabelImg = icvpr_label_color(labelImg)

    NumSet = []

    for i in range(2, len(labels)):
        if len(labels[i]) > 400 or len(labels[i]) < 5:
            labels[i].clear()
            continue

        sort_by_row = sorted(labels[i], key=lambda x: x[0])
        sort_by_col = sorted(labels[i], key=lambda x: x[1])
        x1 = sort_by_row[0][0]
        y1 = sort_by_col[0][1]
        x2 = sort_by_row[-1][0]
        y2 = sort_by_col[-1][1]

        if 110 <= x1 <= 196 and x2 - x1 >= 4 and 10 <= y2 - y1 <= 150:
            cv2.rectangle(colorLabelImg, (y1, x1), (y2, x2), (0, 0, 255))
            NumSet.append(((y1, x1), (y2, x2)))


    NumSet.sort(key=lambda x: x[0][0])

    horizontalbase = NumSet[-1][0][1]
    brackets = []
    bracketnumbers = []
    for i in range(len(NumSet)):
        bracketnumbers.append(NumSet[i])

    print("recognition results of integral")
    lenk = []
    wp = 0
    for i in range(1, len(bracketnumbers)):
        if -5 < bracketnumbers[i - 1][0][0] - bracketnumbers[i][0][0] < 5:
            wp += 1
        else:
            lenk.append(wp)
            wp = 0
    lenk.append(wp)

    Singlenum = []
    count1 = 0
    count2 = 0
    for i in range(len(bracketnumbers)):
        Singlenum.append(bracketnumbers[i])
        count2 = lenk[count1]

        if len(Singlenum) == (count2 + 1):
            Singlenum.sort(key=lambda x: x[0][0], reverse=True)
            Singlenum.sort(key=lambda x: x[0][1], reverse=True)
            num_mat = []
            for k in range(count2 + 1):
                temp = grayImage[Singlenum[k][0][1]:Singlenum[k][1][1] + 1, Singlenum[k][0][0]:Singlenum[k][1][0] + 1]
                temp = cv2.transpose(temp)
                temp = cv2.flip(temp, 1)
                cv2.imwrite("temp.png", temp)
                temp = cv2.imread("temp.png")
                num_mat.append(temp)

            detectnum = ocr.DetectNumero(num_mat)
            numstr = detectnum[:count2 - 1] + "." + detectnum[count2 - 1:count2]
            numbers.append(numstr)

            Singlenum.clear()
            count2 = 0
            count1 += 1

    cv2.imwrite("color.bmp", colorLabelImg)

def convert_to_actual_bracket(scale, bracket_pos):
    actual_bracket_pos = []
    for pos in bracket_pos:
        left = (pos[0] - scale[0]) / scale[2] * scale[3] + scale[1]
        right = (pos[1] - scale[0]) / scale[2] * scale[3] + scale[1]
        actual_bracket_pos.append((left, right))
    return actual_bracket_pos

