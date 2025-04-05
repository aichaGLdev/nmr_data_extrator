from KNN import KNN
import numpy as np

class OCR:
    def __init__(self):
        self.knn_instance = KNN()  # Créer une instance de la classe KNN

    def DetectNum(self, inMats):
        result = ""
        first_char_added = False  # Variable de contrôle pour vérifier si le premier caractère a déjà été ajouté

        for mat in inMats:
            if mat.shape[0] <= 7 and mat.shape[1] <= 8:
                result += "."
            else:
                if not first_char_added:  # Si le premier caractère n'a pas encore été ajouté
                    result += str(self.knn_instance.KNNTest(mat))  # Ajouter le premier caractère détecté
                    first_char_added = True  # Mettre à jour la variable de contrôle
                else:
                    result += "."  # Ajouter un point après le premier caractère
                    result += str(self.knn_instance.KNNTest(mat))  # Ajouter les caractères suivants sans point

        return result

    def DetectNumero(self, inMats):
        result = ""
        for mat in inMats:
            if mat.shape[0] <= 7 and mat.shape[1] <= 8:
                result += "."
            else:
                result += str(self.knn_instance.KNNTest(mat))
        return result

