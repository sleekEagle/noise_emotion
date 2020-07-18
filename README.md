# Emotion recognition under noise

This project deals with spoken emotion recognition under noisy conditions. Training and evaluation are done for
Berlin Database of Emotional Speech and RAVDESS dataset. 
The models and audio data can be found in the directory data_and_models.

## Berlin Database of Emotional Speech
10 models were trained for each fold of the dataset. 
1st fold is the first 0.1 data from the file data_and_models/emodb_data.pkl

Our model hyper parameters were tuned with this dataset.
Trained models can be found data_and_models/emodb_models/

Note there is are 10 models; one nodel per each fold.

## RAVDESS
The model with hyperparameters found from Berlin Database of Emotional Speech was used to train and test on RAVDESS.
The model was trained with 80% of the data from data_and_models/RAVDESS_data.pkl and tested on the rest. 
Trained Keras model is data_and_models/RAVDESS_model.h5

## References

[1] Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song 
(RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. 
PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.

[2] F. Burkhardt, A. Paeschke, M. Rolfes, W. F. Sendlmeier, B. Weiss, A database of German emotional speech,
in: Ninth European Conferenceon Speech Communication and Technology, 2005

Comming up : (Our paper)
