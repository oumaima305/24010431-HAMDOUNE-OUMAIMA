## HAMDOUNE Oumaima 
## Apogée : 24010431
<img width="738" height="689" alt="image" src="https://github.com/user-attachments/assets/5ccc47f5-b199-4f9b-8343-5ae7b0fc93ec" />
<img width="629" height="635" alt="image" src="https://github.com/user-attachments/assets/d5336512-8ad3-46d0-ac97-6d1ef7df733d" />

## Détection automatique des tumeurs cérébrales par apprentissage profond
## ## Développement d'un système de classification automatique d'images IRM pour la détection de tumeurs cérébrales par apprentissage profond
## 1. Contexte général et introduction

La détection des tumeurs cérébrales constitue une étape cruciale dans le diagnostic médical. Elle repose traditionnellement sur l’analyse manuelle des images IRM par des radiologues. Bien que cette méthode soit fiable, elle présente plusieurs limites liées à l’intervention humaine. Avec l’évolution du Machine Learning et plus particulièrement du Deep Learning, il devient possible de développer des systèmes intelligents capables d’analyser automatiquement ces images afin d’assister les professionnels de santé.

Ce travail s’inscrit dans cette perspective et vise à exploiter les réseaux de neurones convolutifs (CNN) pour la classification automatique d’images IRM du cerveau.

## 2. Problématique

La détection manuelle des tumeurs cérébrales sur les images IRM présente plusieurs limitations majeures :

Variabilité inter-opérateur dans l’interprétation des images.

Fatigue des radiologues due à la charge de travail élevée.

Temps d’analyse conséquent pour chaque examen.

Risque d’erreur humaine face à la complexité des images IRM.

Ces contraintes justifient la mise en place d’un système automatique, capable de fournir une aide fiable, rapide et reproductible au diagnostic médical.

## 3. Objectifs du projet

L’objectif principal de ce projet est le développement d’un système de classification automatique d’images IRM pour la détection de tumeurs cérébrales par apprentissage profond.

## Objectifs spécifiques :

Automatiser l’analyse des images IRM.

Classer les images en deux catégories : présence ou absence de tumeur.

Réduire le temps de diagnostic.

Améliorer la précision et la cohérence des résultats.

Fournir un outil d’aide à la décision pour les radiologues.

## 4. Présentation et source de la base de données

Le dataset utilisé est composé d’images IRM du cerveau, organisées en deux classes :

Yes (Brain Tumor) : images contenant une tumeur.

No (No Brain Tumor) : images saines.

## Caractéristiques du dataset :

<img width="738" height="689" alt="image" src="https://github.com/oumaima305/24010431-HAMDOUNE-OUMAIMA/blob/main/Capture%20%C3%A9.PNG?raw=true" />

Les données proviennent d’une plateforme scientifique ouverte (telle que Kaggle), où elles sont mises à disposition par des chercheurs et institutions médicales à des fins pédagogiques et de recherche, après anonymisation.

## 5. Méthodologie de travail

La méthodologie suivie respecte le cycle de vie classique d’un projet de Machine Learning :

Collecte des données : chargement des images IRM.

## Prétraitement :

Redimensionnement des images en 128×128 pixels.

Normalisation des valeurs des pixels entre 0 et 1.

Conversion des images en tableaux NumPy.

Encodage des classes (1 : tumeur, 0 : absence de tumeur).

## Séparation des données :

80 % pour l’entraînement.

20 % pour le test.

## Modélisation : 

conception d’un réseau neuronal convolutionnel (CNN).

## Entraînement :

optimisation des paramètres du modèle.

## Évaluation :

analyse des performances à l’aide de métriques et de graphes.

## 6. Modèle utilisé : Réseau de neurones convolutif (CNN)

Le modèle choisi est un CNN, particulièrement adapté au traitement d’images.

## Architecture générale :

Couches de convolution : extraction des caractéristiques visuelles (formes, contours, textures).

Couches de MaxPooling : réduction de la dimension spatiale.

Couche Flatten : transformation des cartes de caractéristiques en vecteur.

Couches Dense : classification finale.

Fonction d’activation Sigmoid : sortie binaire.

## Justification du choix du CNN :

Excellente performance en vision par ordinateur.

Capacité d’apprentissage automatique des caractéristiques.

Robustesse face aux variations d’images.

## 7. Entraînement du modèle

Le modèle est entraîné avec :

Binary Crossentropy comme fonction de perte.

Adam Optimizer pour l’optimisation.

Taille de batch : 32.

Nombre d’époques : défini selon le code.

L’entraînement permet au modèle de distinguer progressivement les images tumorales des images normales.

## 8. Interprétation du code et des graphes
## 8.1 Graphe de la précision (Accuracy)

Ce graphe illustre l’évolution de la précision sur les ensembles d’entraînement et de validation.

Objectif : vérifier la capacité du modèle à apprendre correctement.

Interprétation : une précision croissante indique un apprentissage efficace.

## 8.2 Graphe de la perte (Loss)

Il représente l’évolution de l’erreur du modèle au fil des époques.

Pourquoi ce graphe : détecter le sous-apprentissage ou le sur-apprentissage.

Interprétation : une perte décroissante indique une amélioration du modèle.

## 8.3 Matrice de confusion

La matrice de confusion compare les prédictions aux valeurs réelles.

Objectif : analyser les erreurs de classification.

Intérêt : identifier les faux positifs et faux négatifs.

 ## 9. Matrice de corrélation

La matrice de corrélation permet d’étudier les relations entre différentes variables ou caractéristiques extraites.

Elle aide à comprendre les dépendances entre certaines caractéristiques.

Une forte corrélation indique une relation importante entre deux variables.

## 10. Lien entre le projet et le Machine Learning

Ce projet illustre une application concrète du Machine Learning supervisé et du Deep Learning :

Utilisation de données étiquetées.

Apprentissage automatique à partir des données.

Amélioration des performances grâce à l’entraînement.

Le Machine Learning permet ainsi de transformer des images IRM brutes en informations exploitables pour l’aide au diagnostic médical.

## 11. Résultats et interprétation globale

Les résultats montrent que le modèle est capable de distinguer efficacement les images avec et sans tumeur, avec une précision généralement élevée. Cela confirme la pertinence de l’approche par apprentissage profond pour ce type de problématique médicale.

## 12. Limites du travail

Malgré les résultats satisfaisants, certaines limites subsistent :

Taille limitée du dataset.

Risque de sur-apprentissage.

Difficulté d’interprétation complète des décisions du modèle.

Généralisation limitée à d’autres bases ou appareils IRM.

## 13. Recommandations et pistes d’amélioration

Plusieurs améliorations peuvent être envisagées :

Utilisation du Transfer Learning (VGG, ResNet, EfficientNet).

Augmentation des données (Data Augmentation).

Optimisation des hyperparamètres.

Intégration de techniques d’explicabilité (Grad-CAM).

Validation sur des données cliniques réelles.

## 14. Conclusion générale

Ce travail démontre que l’apprentissage profond constitue une solution efficace pour la détection automatique des tumeurs cérébrales à partir d’images IRM. Le système développé ne remplace pas le radiologue, mais agit comme un outil d’aide à la décision, contribuant à améliorer la rapidité, la fiabilité et la précision du diagnostic médical.
