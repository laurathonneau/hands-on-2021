# Projet hands-on-2021

Ce projet a pour but de créer une application capable de prédire des panneaux routiers grace à un réseau de neuronnes profonds. 
Pour ce faire nous avons entrainé un modele de réseau de neuronnes, grace à des données de panneaux Allemands, que nous avons pas la suite testé. Ce modèle a ensuite été comparé à un modèle SVM mais celui-ci s'est avéré moins performant, même après un tunning des paramètres. 

Enfin, nous avons crée une application Dash permettant de rentrer une image qui sera ensuite prédite avec le modèle de réseaux de neuronnes. 

## Récapitulatif des differents dossiers

* Script : permet de télécharger les données
* Notebook : comporte le Notebook des modèles de Neural Network et SVM ('train_models_NN_SVM.ipynb') + son fichier yaml associé
* app : comporte le fichier 'app.py' de l'application + un fichier coportant les classes des panneaux + fichier yaml associé à  l'app
* models : comporte le modèle de Neural Network ('traffic_signs_2021-03-19_13-51-00.h5') + le fichier du modèle de SVM ('traffic_signs_svm_final.h5'), tout deux pouvant être reload 


