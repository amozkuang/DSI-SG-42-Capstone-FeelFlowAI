## <img src="https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png" alt="GA logo" style="float: left; margin-right: 10px;" width="50"/>  <img src="img/ff.png" alt="FeelFlow logo" style="float: left; margin-right: 10px;" width="50"/>
# FeelFlow AI: Decoding Emotions, Advancing Patient Support

## DSI-SG-42 Capstone Project

---

## Executive Summary

To enhance diagnosis and treatment outcomes in mental health, our project leverages Speech Emotion Recognition technology to predict patients' emotional states accurately. This initiative, supported by the Innovation Team at AI Singapore, targets clinicians, therapists, and mental health professionals attending the "Mental Health with AI" seminar organized by the Ministry of Health.

Our process began with scraping unseen YouTube audio data, where GenZ and millennials openly discuss their mental health struggles. We then pooled training and testing data from data that are commonly used in the Speech Emotion Recognition space (CREMA-D, TESS, ESD). The data underwent preprocessing, including audio augmentation and feature extraction, and was then combined with training and test data for comprehensive analysis. We employed multiple models—Random Forest, MLP, LSTM + 1D CNN, LSTM + 2D CNN, and WaveNet—to model seen data, selecting WaveNet for its superior performance (Training Accuracy: 0.65, Validation Accuracy: 0.61) on unseen data, validated using ESD data.

However, the application faces challenges like cultural and linguistic nuances, the complexity of emotional spectrum, and the need for real-time processing capabilities. To address these, we recommend expanding the dataset to include a wider range of emotions and integrating to train on local speech corpus.

Nonetheless, by implementing these measures, along with ongoing user training and technical support, this tool aims to bridge the gap in understanding emotional cues from younger demographics, thus enabling more targeted and effective mental health interventions in Singapore's diverse setting. 

## Problem Statement

### *Where discerning people’s emotion can sometimes be an unnerving guessing game. How can clinicians use speech emotion recognition technology to accurately assess patients' emotional well-being, thereby improving diagnosis and treatment outcomes?*

## Objectives
The objectives of this study is build an based on pre-trained models to accurate predict the emotional state through the patients’ speech. Thereby, alleviating a layer of complexity, which allows the clinician to focus on the diagnosis and/or treatment.

 

## Success Metric

1. Number of people emotions recognised correctly
2. Within each emotion , how many times how many times do accurate predictions occur

## Definitions
### Categories
Since we're looking to predict, we narrowed in on 6 emotions (taking after CREMA-D):
* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad

## The Data 
The datasets used for this study can be referred to as - Seen & Unseen data. Each audio file is about 2-3 seconds each, and have all been processed with the same measures.

Due to the capacity of storage required (>10GB), we will not be uploading onto this repository. Instead, please find the datasets downloadable [here](https://drive.google.com/drive/folders/16Vfxe5vMqHd7-qpxqs9QpbE6DAvY3Hbe?usp=sharing).

1) Seen Data
    - [Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)](https://github.com/CheyneyComputerScience/CREMA-D)

    - [Toronto Emotional Speech Set (TESS)](https://tspace.library.utoronto.ca/handle/1807/24487)

    - [Emotion Speech Dataset (ESD)](https://github.com/HLTSingapore/Emotional-Speech-Data)

2) Unseen Data
    - YouTube data (consisting of 8 videos documenting people sharing their mental health struggles/experience): This data is not publicly known to have been trained on before. We scrutinized the video to only include segments where conversations on the subject's sharing of their mental health journey/experiences are included. In sharing, some of them relive the pain and emotions, which we can then further analyse as part of data.

    - [Emotion Speech Dataset (ESD)](https://github.com/HLTSingapore/Emotional-Speech-Data): From the parent dataset above, we split the evaluation data out to use as a medium of validating the model's predictability.


## Requirements for Code Execution
For the purpose of this project, we'll be alternating between 2 Python version(s). Reason for this is due to dependency issues with the different packages required
[Note: It is advisable to run Python 3.9.6 for notebooks [#1](code/01_Scraping%20(YouTube%20audio).ipynb) & [#2](code/02_Preprocessing.ipynb), while the rest is suited to run on Python 3.8.19]. 

With that would require the installation of specific libraries/package versions. Please find link to the requirements for the different Python versions.

1. [Python 3.8.19](requirements_3.8.txt)
2. [Python 3.9.6](requirements_3.9.txt)

---
