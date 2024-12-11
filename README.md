<div align="center">

# Code for EEG-Music Emotion Recognition Grand Challenge 

</div>

# Overview
Methodology implementation for the <a href='https://eeg-music-challenge.github.io/eeg-music-challenge/'>EEG-Music Emotion Recognition Grand Challenge</a> hosted at <a href='https://2025.ieeeicassp.org/sp-grand-challenges/#gc5'>ICASSP2025</a>.

# Methods

## Preprocessing
We used pruned data as starting point and we applied minimal preprocessing involving:
- deletion of invalid values (nans and outliers)
- standardization with global per-channel statistics

## Feature Extraction
We extracted Connectivity Feature Maps (CFM) of the preprocessed EEG signals for the Emotion Recognition Task.

## Architectures
We chose following architectures as backbone :
- EEGConformer [1]
- 2D CNN [2]

Bi-LSTM based Projection network was added on top of EEGConformer for the subject identification task while on top of 2D CNN we added auxiliary classification head in addition to typical single classification head for the emotion recognition task.


## Training
For the subject identification task, we created a small validation set (val_trial) by extracting 2 trials per subject from the training data. For the emotion recognition task, before extracting the held-out-trial validation set, we selected 2 subjects to serve as a separate held-out-subject validation set (val_subject).

Models were trained using the Adam optimizer for 500 epochs. During training, the model was provided with a random window of 1280 timepoints. For validation, we first segmented each sample into smaller windows of 1280 timepoints, excluding the final segment. The model was then fed all the windows, and a voting scheme (average for subject identification and majority for emotion recognition) was applied to determine the final prediction.
While training on emotion recognition task,CFM of whole EEG signal was taken out, no windowing was applied.



## Inference
For inference, same as for validation, each sample was first segmented into smaller windows of 1280 timepoints, excluding the final segment. 
The same voting scheme applied in validation was used to generate the final prediction, but no windowing of CFMs for emotion recognition task.

# Results
Our strategy yields the following results

| Model             | Subject Identification | Emotion Recognition |
|-------------------|------------------------|---------------------|
| EEGConformer         | 97.49%                  | -               |
| 2D CNN      | -                  | 38.76%               |



# How to run

### **Requirements**

- Download dataset from the <a href='https://kaggle.com/datasets/e25d8f6d371bfbe7f35f67458a7759de80d809f970f33b05ff22e7abb70bd65a'>EREMUS Kaggle page</a> 
- Place your dataset where you prefer and change the key `dataset_path` on `config.json` file accordingly 
- Create a conda environment through `conda env create -n emer --file environme
nt.yml`
- Optionally, create a wandb account and change the key `wandb_entity` on `config.json` file accordingly. 

All the baselies were tested on a single NVIDIA RTX A6000 GPU.

### **Preprocess data**

```
python preprocess.py --split_dir data/splits
```

### **Train a model for subject identification**

```
python train.py --task subject_identification --model eegnet --lr 0.001 --epochs 100
```

Model weights at best validation accuracy will be saved at exps/subject_identification

### **Train a model for emotion recognition**

```
python train.py --task emotion_recognition --model eegnet --voting_strategy majority --lr 0.001 --epochs 100
```

Model weights at best validation accuracy will be saved at exps/emotion_recognition

### **Inference**

This script will generate te required file for the final submission.
Always specify:
- the task 
- the model architectures
- the path (absolute or relative) to the folder with .pth file

As an example you can run:

```
python inference.py --task subject_identification --model eegnet --voting_strategy mean --resume exps/subject_identification/eegnet/baseline_2024-08-29_16-17-27
```

Running inference on **subject identification** will create a csv file named *results_subject_identification_test_trial.csv* for the held-out-trial test set.

Running inference on **emotion recognition** will create two csv files:
- *results_emotion_recognition_test_trial.csv* for the held-out-trial test set.
- *results_emotion_recognition_test_subject.csv* for the held-out-subject test set.

Each csv has only two columns:
- **id**: the id of the sample
- **prediction**: the predicted class
