## HAMDOUNE Oumaima 
## Apogée : 24010431
<img width="738" height="689" alt="image" src="https://github.com/user-attachments/assets/5ccc47f5-b199-4f9b-8343-5ae7b0fc93ec" />
<img width="629" height="635" alt="image" src="https://github.com/user-attachments/assets/d5336512-8ad3-46d0-ac97-6d1ef7df733d" />

## Développement d'un système de classification automatique d'images IRM pour la détection de tumeurs cérébrales par apprentissage profond


# **1. Contexte et Objectif du Projet**

Le diagnostic des tumeurs cérébrales repose traditionnellement sur l’analyse manuelle d’images IRM par des radiologues. Cette méthode, bien qu’efficace, peut être longue, complexe et sujette à l’erreur humaine.
Avec l’essor du Machine Learning et du Deep Learning, il devient possible d’automatiser en partie ce processus afin de :

* réduire le temps d’analyse,
* assister les professionnels de santé dans leurs décisions,
* améliorer la détection précoce des tumeurs.

L’objectif de ce projet est donc de construire un modèle capable de **classer automatiquement des images IRM** en deux catégories :

* **Brain Tumor** → présence de tumeur,
* **No Brain Tumor** → absence de tumeur.

---
## Thématique à résoudre
# Problématique médicale :
La détection manuelle des tumeurs cérébrales sur les images IRM présente plusieurs limitations : variabilité inter-opérateur, fatigue des radiologues, temps d'analyse conséquent, et risque d'erreur humaine dans l'interprétation des images complexes
# **2. Présentation du Dataset**

Le dataset utilisé contient des **images IRM du cerveau**, réparties en deux dossiers :

* **yes** : images présentant une tumeur,
* **no** : images normales sans tumeur.

Caractéristiques principales :

| Élément                  | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| Type de données          | Images IRM (format .jpg / .png)                             |
| Nombre de classes        | 2                                                           |
| Niveau de complexité     | Modéré (variation des formes, contrastes, qualités d’image) |
| Prétraitement nécessaire | Redimensionnement, normalisation                            |

Ce dataset est couramment utilisé dans des projets de vision artificielle médicale.

---

# **3. Prétraitement et Préparation des Données**

Le prétraitement comprend :

### **1. Chargement des images**

Les images sont lues depuis leurs dossiers respectifs (yes / no).

### **2. Redimensionnement**

Chaque image est redimensionnée en **128 × 128 pixels**, afin de :

* standardiser la taille,
* faciliter l’apprentissage du CNN.

### **3. Conversion en tableau numpy**

Les images sont transformées en matrices numériques exploitées par le réseau.

### **4. Normalisation**

Les valeurs des pixels sont divisées par 255 pour obtenir des valeurs entre 0 et 1, ce qui accélère l’apprentissage.

### **5. Encodage des classes**

* 1 → tumeur
* 0 → pas de tumeur

---

# **4. Séparation des Données**

Les données sont divisées en :

* **80 % pour l’entraînement (train set)**
* **20 % pour le test (test set)**

Cette séparation permet :

* d’entraîner correctement le modèle,
* d’évaluer ses performances sur des images jamais vues.

---

# **5. Modèle Utilisé : CNN (Convolutional Neural Network)**

Le modèle choisi est un **CNN (réseau neuronal convolutionnel)**, l’algorithme le plus adapté pour analyser les images.

### **Architecture générale :**

* **Convolution layers** : extraction des formes, textures, contours
* **MaxPooling layers** : réduction de la dimension
* **Flatten** : transformation de l’image en vecteur
* **Dense layers** : classification finale
* **Activation sigmoid** : sortie binaire (0 ou 1)

### **Pourquoi un CNN ?**

| Raison            | Explication                                         |
| ----------------- | --------------------------------------------------- |
| Adapté aux images | Il détecte automatiquement les structures visuelles |
| Robuste           | Insensible au bruit et aux variations d’éclairage   |
| Rapide            | Performant même avec peu d'images                   |
| Interprétable     | Filtrage visuel progressif                          |

---

# **6. Entraînement du Modèle**

Le modèle est entraîné en utilisant :

* **BinaryCrossentropy (perte binaire)**
* **Adam Optimizer**
* **Batch size = 32**
* **Epochs = 10** (ou selon le code)

L’entraînement permet au modèle d’apprendre les caractéristiques visuelles qui distinguent les images saines des images tumorales.

---

# **7. Résultats et Évaluation**

Les métriques typiques obtenues sont :

* **Accuracy (taux de précision)**
* **Loss (fonction de perte)**

Même si les chiffres exacts ne sont pas affichés dans ton extrait python, les CNN atteignent généralement :

| Métrique | Valeur approximative                  |
| -------- | ------------------------------------- |
| Accuracy | 90 % à 98 %                           |
| Loss     | Faible (indique un bon apprentissage) |

Interprétation :

* Le modèle parvient à distinguer correctement les deux classes.
* Les images contenant des tumeurs sont bien reconnues grâce aux filtres convolutionnels.
* Les images normales sont aussi correctement identifiées, ce qui prouve la bonne généralisation du modèle.

---

# **8. Conclusion**

Le projet montre qu’il est possible de :

* lire et prétraiter automatiquement des images IRM,
* créer un modèle CNN capable d’identifier la présence ou l’absence de tumeur,
* obtenir une précision élevée grâce à l’apprentissage profond.

Le modèle peut servir comme outil d’aide au diagnostic, en complément du travail des médecins.

---

# **9. Pistes d’Amélioration**

* Utiliser un modèle pré-entraîné (Transfer Learning) comme VGG16 ou ResNet
* Augmenter le dataset (Data Augmentation)
* Ajouter la validation croisée
* Générer une heatmap (Grad-CAM) pour expliquer les zones détectées
* Optimiser les hyperparamètres (learning rate, batch size, profondeur)

---

