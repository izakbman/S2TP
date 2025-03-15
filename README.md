# S2TP Automatic Speech Recognition (ASR) Project

## Introduction

Automatic Speech Recognition (ASR) is a critical field of deep learning that powers many speech-to-text applications, such as captioning tools and transcription services. It is essential for a wide range of use cases, including accessibility tools for the deaf and hard-of-hearing community, as well as for services like customer service call transcripts, audio summaries, and more.

### ASR's Impact on the Deaf Community
ASR is especially important for individuals who rely on captioning as a means of communication. As someone who grew up with a deaf father, I have personally witnessed the importance of these tools. Over the years, I’ve seen captioning tools, like Ava, Otter.ai, and Google Captions, evolve, offering higher-quality transcriptions and benefiting from continuous improvements in ASR technology.

### Project Motivation
The motivation for this project stemmed from my personal experiences with captioning tools and the advancements I've observed in ASR technology. The goal of this project was to explore the deep learning techniques driving ASR systems, evaluate how well my own model performs, and compare it against a pretrained ASR model to better understand their effectiveness in real-world scenarios.

## Data Description

#### LibriSpeech ASR Dataset
The LibriSpeech dataset is a widely used collection in the field of Automatic Speech Recognition (ASR). It consists of up to 1000 hours of spoken word audio from audiobooks, which is provided by OpenSLR and can be downloaded from Hugging Face (HF). The dataset is segmented into approximately 5-second audio chunks for training and evaluation purposes.

### Dataset Details
* Train Split: 2162 samples
* Validation Split: 270 samples
* Test Split: 271 samples
**Each example in the dataset contains:**

* .flac file: The raw audio source.
* Audio Array: A vectorized form of the raw audio (e.g., similar to what is plotted in the mel-spectrogram).
* Speaker ID: Identifies the speaker in the audio sample.
* Chapter ID: The chapter identifier from the audiobook.
* Metadata: Includes details about audio quality and the sampling rate.
For this project, I manually created subsets of the validation split to tailor the data for training and evaluation.


!["Model Structure"](images/speech-recognition-1024x576.webp)

## Features

data_utility.py: download and store librosa asr dataset from hugging face to librosa_cache directory
model.py: train and evaluate my simpleCTCmodel and Wav2Vec from meta
notebook.ipynb: cells to organize and run project code 


## Usage 

#### Model.py
**Args**
* --mymodel: run training and evaluation code for simpleCTC Model, model weights saved to evaluation dir
* --wav2vec: evaluate training data with wav2vec model (meta)
* adding both args will compare the two models
* adding --eval will only evaluate simpleCTC model and save evaluation plots to evaluation dir
  * if there are no model weights saved user will be prompted to train the model
 
*please note... my model is currently suffering from vaninishing/exploding gradients resulting in nan loss*

*--eval will evaluate based on saved (nonsensical weights) code may not work as expected*

#### data_utility.py
**Args** 

 * --talapas: save librispeech validation split to librispeech_cache dir 
 
 * --display: display transcription, raw wavefom, mel spectogram and playable audio

