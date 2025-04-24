# Bird Species Classifier from Audio using CNN and EfficientNet_B0 (AMLS-II Project)

This project implements two deep learning models for classifying bird species from audio clips: a custom convolutional neural network (CNN) and a transfer learning approach using EfficientNet_B0. Audio files are converted into Mel-spectrogram images and classified into one of 24 bird species.

## Project Structure
 - main.py , Main File, running it will train and evaluate both models.
 - misc.ipynb , Same code as main.py but in notebook format for easier trialing initially.
 - A/ , All Source Code:
    + cnn_model.py , Custom CNN model Code
    + data_preproc.py , Contains functions that handle data preprocessing
    + spec_gen.py , Contains code that converts sound files to mel spectrograms while maintaining folder structure
    + trans_model.py , Transfer Learning Model Code
    + utils.py , Contains code for training, testing, and plotting loss curves and confusion matrices
- Datasets/
    + Sound-Files , Contains the entire dataset in the form of .ogg sound files grouped by species
    + Spec-Images , Contains the entire dataset in the form of mel spectrogram images grouped by species
- Results , Used for storing loss curves and confusion matrices pics
- env/
    + environment.yml , Code to create a new conda env called "BirdCLEF" with all necessary modules
    + requirements.txt, All needed modules
- README.md , This file

## All Packages needed to run the code
- numpy
- pandas
- matplotlib
- torch
- torchvision
- tqdm
- librosa
- scikit-learn
- timm

## Instructions
- git clone the repo, go into the env folder and open terminal
- run "conda env create -f environment.yml" to create "BirdCLEF" environment with all necessary modules
- run main.py to train both models, obtain loss plots, accuracy figures and confusion matrices (takes around 1 hour on my laptop with Nvidia GPU) 