from FileOp import FileOp
import cv2  # Importation de la bibliothèque OpenCV
import os  # Importation du module os pour la manipulation de fichiers et de répertoires
import numpy as np  # Importation de la bibliothèque NumPy pour le traitement des tableaux
class KNN:
    def __init__(self):
        self.knn = cv2.ml.KNearest.create() # Initialisation du classificateur KNN

    def KNNTrain(self, trainpath):  # Méthode pour l'entraînement du modèle KNN
        trainingImages = []
        trainingLabels = []
        fileop = FileOp()
        fileop.getTrainSet(trainpath, trainingImages, trainingLabels)  # Obtenir les données d'entraînement
        trainingData = np.array(trainingImages, dtype=np.float32).reshape(-1, 648) # Convertir les données en tableau NumPy de type float32
        classes = np.array(trainingLabels)  # Convertir les étiquettes en tableau NumPy
        if trainingData.size == 0 or classes.size == 0:  # Vérifier si les données d'entraînement sont vides
            print("Error: trainingData or classes is empty")
            return
        num_samples = trainingData.shape[0]  # Nombre d'échantillons dans les données d'entraînement
        num_features = trainingData.shape[1]  # Nombre de caractéristiques dans chaque échantillon
        print(f"Number of samples: {num_samples}")
        print(f"Number of features: {num_features}")
        print("Shape of trainingData:", trainingData.shape)
        print("Size of trainingData:", trainingData.size)
        print("Type of trainingData:", trainingData.dtype)

        self.knn.setDefaultK(4)  # Définir le nombre de voisins par défaut à 4
        try:
            self.knn.train(trainingData, cv2.ml.ROW_SAMPLE, classes)  # Entraîner le classificateur KNN
            print("Entraînement KNN terminé !")
        except cv2.error as e:
            print("Error during KNN training:", e)

    def KNNTest(self, inMat):
        inMat = cv2.resize(inMat, (12, 18))
        p = inMat.reshape(1, -1).astype(np.float32)
        _, result, _, _ = self.knn.findNearest(p, k=1)
        return int(result)

    def KNNTestSet(self, testsetpath):
        # Obtenir la liste des noms de fichiers d'images de test
        filenames = []
        fileop = FileOp()
        fileop.getFiles(testsetpath, filenames)

        # Chemin de sauvegarde pour les images de test
        save_path = "knntestset/"

        # Taille totale de la liste des fichiers
        wholesize = len(filenames)

        # Initialiser le compteur pour les prédictions correctes
        correctsize = 0

        # Boucler à travers chaque fichier
        for i, filename in enumerate(filenames):
            # Extraire le numéro de classe de l'image
            num = int(filename.split('_')[1].split('.')[0])

            # Charger l'image de test
            inMat = cv2.imread(filename)
            inMat = cv2.resize(inMat, (12, 18))  # Redimensionner l'image en 12x18 pixels
            inMat = inMat.astype(np.float32)
            # Prédire la classe de l'image à l'aide du modèle KNN
            result = self.KNNTest(inMat)
            # Construire le nom de fichier de sauvegarde
            savename = os.path.join(save_path, f"{i + 1}_{result}.png")
            # Enregistrer l'image de test
            cv2.imwrite(savename, inMat)

            # Vérifier si la prédiction est correcte et mettre à jour le compteur
            if result == num:
                correctsize += 1
            else:
                print("Erreur de prédiction pour l'image:", filename)

        # Calculer le taux de précision de la classification
        accuracy = correctsize / wholesize
        print("Taux de précision:", accuracy)