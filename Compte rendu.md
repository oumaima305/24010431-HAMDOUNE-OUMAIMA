# Compte Rendu d'Analyse Croisée de Deux Projets de Science des Données en Imagerie Médicale

Ce rapport a pour objectif de synthétiser et de mettre en perspective les informations contenues dans les deux documents fournis : un guide théorique et pratique sur la classification du cancer du sein (`CorrectionProjet.md`) et un script d'implémentation d'un réseau neuronal pour la détection de tumeurs cérébrales (`Untitled9.ipynb`).

Bien que les deux projets relèvent du domaine de l'aide au diagnostic médical par l'intelligence artificielle, ils illustrent deux approches distinctes de la science des données : le Machine Learning Classique sur données tabulaires et le Deep Learning sur données d'imagerie.

---

## 1. Projet 1 : Classification du Cancer du Sein (Analyse Tabulaire)

Le document `CorrectionProjet.md` est un guide pédagogique qui décortique un projet de classification binaire basé sur le *Breast Cancer Wisconsin Dataset*. L'accent est mis sur la **compréhension des mécanismes internes** et la **traduction des contraintes métier en choix techniques**.

### A. Le Contexte Métier et la Contrainte Critique

Le point de départ de ce projet est la nécessité de créer un "Assistant IA" pour le second avis médical. La contrainte la plus importante est l'**asymétrie des coûts d'erreur** [1].

> Dire à un patient malade qu'il est sain (Faux Négatif) peut entraîner la mort par retard de traitement. **L'IA doit donc prioriser la sensibilité (Recall).**

Cette exigence métier dicte le choix de la métrique d'évaluation : la **Sensibilité (Recall)**, définie comme $TP / (TP + FN)$, doit être maximisée, quitte à accepter un taux légèrement plus élevé de Faux Positifs (stress et coûts de biopsie).

### B. Méthodologie et Points Techniques Clés

| Phase du Projet | Description | Enjeu Technique |
| :--- | :--- | :--- |
| **Data Wrangling** | Simulation de données manquantes (5% de `NaN`) et imputation par la moyenne (`SimpleImputer(strategy='mean')`). | Le document met en garde contre le **Data Leakage** (Fuite de données), une erreur subtile où l'imputation est effectuée avant la séparation Train/Test, introduisant des informations du futur dans l'entraînement [1]. |
| **Modélisation** | Utilisation d'un **Random Forest Classifier**. | Le Random Forest est choisi pour sa robustesse. Le document explique son fonctionnement par **Bagging** (Bootstrapping) et **Feature Randomness**, permettant de réduire la **haute variance** des arbres de décision individuels [1]. |
| **Évaluation** | Analyse de la Matrice de Confusion et du Rapport de Classification. | L'analyse se concentre sur la distinction entre **Précision** (qualité de l'alarme) et **Rappel** (puissance du filet), soulignant que l'Accuracy est une métrique dangereuse en cas de déséquilibre de classes [1]. |

---

## 2. Projet 2 : Détection de Tumeurs Cérébrales (Deep Learning sur Images)

Le fichier `Untitled9.ipynb` présente l'implémentation d'un projet de **classification d'images** pour la détection de tumeurs cérébrales à partir d'images IRM. Ce projet s'appuie sur une approche de **Deep Learning** utilisant des Réseaux Neuronaux Convolutifs (CNN).

### A. Données et Préparation

Le script est conçu pour charger des images depuis un chemin de dataset (identifié comme `/kaggle/input/brain-mri-images-for-brain-tumor-detection` dans l'exécution observée).

*   **Type de Données :** Images IRM (3 canaux, RGB).
*   **Pré-traitement :** Les images sont redimensionnées à **128x128 pixels** et **normalisées** (division par 255.0) pour ramener les intensités de pixels dans la plage [0, 1].
*   **Distribution :** L'exécution a chargé **253 images** au total, avec un déséquilibre de classes notable : 155 images de classe 'yes' (61.3%) et 98 images de classe 'no' (38.7%) [2].
*   **Protocole Expérimental :** Les données sont divisées en trois ensembles : **Entraînement, Validation et Test** (avec une stratégie de *stratification* pour maintenir la distribution des classes dans chaque split).

### B. Architecture du Modèle CNN

Le modèle est une architecture séquentielle typique de CNN, conçue pour l'extraction automatique de caractéristiques à partir des images :

1.  **Couches Convolutives (`Conv2D`) :** Elles appliquent des filtres pour détecter des motifs (bords, textures). Le modèle utilise deux blocs de convolution (32 et 64 filtres).
2.  **Normalisation par Lot (`BatchNormalization`) :** Stabilise et accélère l'entraînement.
3.  **Pooling (`MaxPooling2D`) :** Réduit la dimensionnalité et rend le modèle plus tolérant aux variations de position.
4.  **Régularisation (`Dropout`) :** Appliquée après chaque bloc pour prévenir le surapprentissage (overfitting).
5.  **Couches Denses (`Dense`) :** Après un aplatissement (`Flatten`), les couches denses effectuent la classification finale. La couche de sortie utilise une activation `sigmoid` pour la classification binaire [2].

Le modèle est compilé avec l'optimiseur **Adam** et la fonction de perte **Binary Cross-Entropy**, standard pour les problèmes de classification binaire.

---

## 3. Synthèse et Contraste des Approches

Les deux documents offrent un aperçu de la diversité des défis en Data Science médicale. Le tableau ci-dessous résume les différences fondamentales entre les deux projets :

| Caractéristique | Projet 1 (Cancer du Sein) | Projet 2 (Tumeur Cérébrale) |
| :--- | :--- | :--- |
| **Type de Données** | Tabulaire (30 caractéristiques extraites) | Imagerie (IRM, pixels bruts) |
| **Méthodologie** | Machine Learning Classique | Deep Learning (CNN) |
| **Modèle Principal** | Random Forest | Réseau Neuronal Convolutif (CNN) |
| **Pré-traitement** | Imputation des valeurs manquantes | Redimensionnement et Normalisation des pixels |
| **Enjeu Métrique** | Maximiser le **Recall** (Sensibilité) | Minimiser l'erreur globale (Accuracy/Loss) |
| **Complexité du Modèle** | Faible à Modérée (Interprétable) | Élevée (Boîte noire) |
| **Objectif** | Classification basée sur des mesures | Classification basée sur des motifs visuels |

### Conclusion

Le premier projet insiste sur la **logique métier** et la **robustesse statistique** (gestion du Data Leakage, choix du Recall), typique des analyses sur données structurées. Le second projet met en évidence la **puissance des CNN** pour l'extraction automatique de caractéristiques complexes à partir de données non structurées (images), où l'accent est mis sur l'architecture du réseau et le pré-traitement des données brutes. Ensemble, ils forment un panorama complet des compétences requises pour un ingénieur en Data Science travaillant dans le domaine de la santé.

***

### Références

[1] CorrectionProjet.md : Guide théorique et pratique sur l'anatomie d'un projet Data Science (Classification du Cancer du Sein).
[2] Untitled9.ipynb : Script Jupyter Notebook pour la détection de tumeurs cérébrales par CNN.


