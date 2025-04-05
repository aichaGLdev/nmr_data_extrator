import cv2  # Importation de la bibliothèque OpenCV
import os  # Importation du module os pour la manipulation de fichiers et de répertoires
import numpy as np  # Importation de la bibliothèque NumPy pour le traitement des tableaux

class FileOp:
    def getFiles(self, path, files):  # Méthode pour obtenir la liste des fichiers dans un répertoire
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                files.append(os.path.join(root, filename))

    def getLabel(self, filename):  # Méthode pour extraire l'étiquette à partir du nom de fichier
        pos = filename.rfind('_') + 1
        label = int(filename[pos])
        return label

    def getTrainSet(self, trainsetpath, trainingImages, trainingLabels):  # Méthode pour obtenir l'ensemble de données d'entraînement
        files = []
        self.getFiles(trainsetpath, files)  # Obtenir la liste des fichiers dans le répertoire spécifié
        for file in files:
            src_image = cv2.imread(file)  # Charger l'image à partir du fichier
            src_image = cv2.resize(src_image, (12, 18))  # Redimensionner l'image en 12x18 pixels
            src_image = src_image.reshape(1, -1)  # Aplatir l'image en un vecteur ligne
            trainingImages.append(src_image)  # Ajouter l'image à la liste des images d'entraînement
            trainingLabels.append(self.getLabel(file))  # Ajouter l'étiquette à la liste des étiquettes d'entraînement
