# AMP--Parkinsons-Disease-Progression-Prediction

## Introduction

Parkinson's disease, affecting over 10 million individuals worldwide, presents challenges in accurate diagnosis and timely treatment. This project leverages Deep Learning techniques to analyze mass spectrometry data of cerebrospinal fluid (CSF) samples, aiming to predict the progression of Parkinson's Disease using Unified Parkinson's Disease Rating Scale (UPDRS) scores.

![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/993a5827-e984-4a83-bb5f-d97a9b0caa6d)

## Dataset

The dataset is sourced from the Kaggle competition 'AMP®-Parkinson's Disease Progression Prediction.' It includes mass spectrometry data at the peptide level, protein expression frequencies, and clinical data with UPDRS scores. The goal is to estimate and predict UPDRS scores for various time intervals.

![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/c904144f-dd94-42f6-99f9-744dd7210e89)

-Train_peptides.csv:  Peptide-level mass spectrometry data, where each peptide is a subunit part of a protein. The file contains details about the month of the visit, relative to the first visit by the patient, and the amount of the peptide and the sequence and frequency of amino acids included in the peptide

-Train_proteins.csv: Protein expression frequencies compiled from peptide level data including visit and patient details, the related protein's UniProt ID code, and normalized protein expression and the frequency of the protein's occurrence in the sample.

-Train_clinical_data.csv: Clinical information on each patient, such as details about the appointment and the patient as well as results from the various sections of the Unified Parkinson's Disease Rating Scale, which rates the severity of PD symptoms,where higher numbers indicate more severe symptoms along with  details like clinical state of mind whether or not the patient was taking medication such as Levodopa during the UPDRS assessment.

-Supplemental_clinical_data.csv:  Clinical data without any corresponding CSF samples that are meant to offer context to the normal development of Parkinson's disease.This information is meant to give further background on how Parkinson's disease typically progresses. similar columns to train_clinical_data.csv are used.

## Project Architecture
![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/8cf35355-8f8d-4d82-a4c9-84b6f2a7493b)

### 1. Data Loading and Preprocessing

The project involves preprocessing clinical data, creating new columns for UPDRS scores at different time intervals, and handling peptide and protein data separately.

### 2. Feature Extraction

Two feature extraction methods are employed using peptide abundance values and normalized protein expression values.

### 3. Data Merging

Preprocessed clinical data is merged with protein and peptide data separately, creating feature matrices 'X' and target matrices 'y' for model training.

### 4. Data Transformation

The feature matrix 'X' is transformed using a column transformer that applies KNN imputation and standard scaling to numerical features, resulting in a transformed feature matrix 'X_transformed.'

### 5. Model Architecture

#### 5.1 Gated Recurrent Unit (GRU) Model

The sequential model with Gated Recurrent Units (GRUs) demonstrates its potential for predicting the progression of Parkinson's disease. The model's ability to capture temporal dependencies in the data can provide valuable insights for healthcare professionals and patients in managing the condition and developing personalized treatment plans.

##### Model Details:

- Three GRU layers (128, 64, and 32 units)
- ReLU activation functions
- L2 regularization, dropout layers (0.15)
- Adam optimizer
- SMAPE+1 loss function
- Trained for 50 epochs, batch size of 64, and 20% validation split

##### Hyperparameter Tuning:

- Activation Functions: ReLU
- Regularization: L2 regularization
- Optimizer: Adam
- Dropout Rate: 0.15
- Epochs: 50
- Batch Size: 64

#### 5.2 SimpleRNN Model

A sequential architecture with SimpleRNN layers has also been used for predicting Parkinson's disease progression.

##### Model Details:

- Three SimpleRNN layers (128, 64, and 32 units)
- ReLU activation functions, L2 regularization
- Dropout layers (0.15)
- Adam optimizer
- SMAPE+1 loss function
- Trained for 50 epochs, batch size of 64, and 20% validation split

##### Hyperparameter Tuning:

- Activation Functions: ReLU
- Regularization: L2 regularization
- Optimizer: Adam
- Dropout Rate: 0.15
- Epochs: 50
- Batch Size: 64

#### 5.3 Sequential Dense Neural Network

A Neural Network model using a sequential architecture with fully connected Dense layers has also been used for predicting Parkinson's disease progression.

##### Model Details:

- Four Dense layers (256, 128, 64, and 32 units)
- ReLU activation functions
- L2 regularization, dropout layers (0.20)
- Stochastic Gradient Descent (SGD) optimizer
- SMAPE loss function
- Trained for 500 epochs, batch size of 32, and 20% validation split

##### Hyperparameter Tuning:

- Activation Functions: ReLU
- Regularization: L2 regularization
- Optimizer: SGD
- Dropout Rate: 0.20
- Epochs: 500
- Batch Size: 32
- 
## 7. Evaluation

The goal of the competition is to predict UPDRS scores for patients with Parkinson's disease at different time points based on protein/peptide samples. The accuracy of the predictions will be evaluated using the Symmetric Mean Absolute Percentage Error (SMAPE) metric.

### SMAPE+1 Metric

The SMAPE+1 metric takes into account the relative error between the predicted and actual UPDRS scores, considering the clinical relevance by weighing the error differently based on the magnitude of the UPDRS score.

#### Best Model Performance

Among all the models, using Unit Protein with GRU as a feature gave the best performance on Kaggle with the following results:

- Training loss: 0.66
- Validation loss: 0.55

The GRU model is configured with three layers (128, 64, and 32 units, respectively), and dropout layers with a rate of 0.15 after each GRU layer to reduce overfitting. An L2 regularization with a coefficient of 0.01 is applied to the second and third GRU layers. The output layer is a dense layer with linear activation. The model is compiled using the 'adam' optimizer, with a custom SMAPE+1 loss function and metric required for the Kaggle competition.

Visualizations of the results for models using Peptides and Unitprot as individual features for different Deep Learning Models can be found in the figures below.
-Using Peptide as features:
  -1.	GRU
  ![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/2757cb2c-55dc-4da2-a36f-a1dc8f527cd7)

  -2.	Deep NN Model
  ![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/11b72c0a-29ae-4ba6-ba3e-4bb0ee9475d5)

  -3.	RNN
  ![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/be1f4882-7916-4347-83d0-18f15e224a07)

-Using UnitProt as features
  -1.	GRU
  ![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/e09d2058-e2ad-4c01-9ba8-260fbcb6aecd)

  -2.	Deep NN model
  ![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/57d49a99-be6e-449c-bd83-bb2d04ff25b7)

  -3.	RNN
  ![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/8602f01d-aef0-40bd-bd77-1d3ce3b3eb1b)


## Competition Rankings

As a participant in the Kaggle competition, our team is currently ranked at 258 out of 1677 teams, putting us in the top 16 percent of all teams. We continuously work to refine our models and explore new approaches to improve our performance in the competition.

[Kaggle Competition Overview](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/overview)

![image](https://github.com/Lohitha-Vanteru/AMP--Parkinsons-Disease-Progression-Prediction/assets/113141006/0c935f5a-07cd-4f7d-b635-439061fe74fa)

## Acknowledgements

The dataset is sourced from the Kaggle competition 'AMP®-Parkinson's Disease Progression Prediction.'

If you use this dataset, please acknowledge Kaggle and the competition organizers.

## License

This project is licensed under the [MIT License](LICENSE).
