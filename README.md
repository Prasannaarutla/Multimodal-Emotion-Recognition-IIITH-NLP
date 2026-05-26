Multimodal Emotion Recognition System
Project Overview

This project implements a Multimodal Emotion Recognition System using both speech and text modalities. The system predicts human emotions by combining information extracted from audio speech signals and corresponding text transcripts.

The project uses the Toronto Emotional Speech Set (TESS) dataset and performs:

Speech Emotion Recognition using MFCC + BiLSTM
Text Emotion Recognition using BERT
Multimodal Fusion of Speech and Text Predictions

The final system predicts emotions such as:

Angry
Disgust
Fear
Happy
Neutral
Pleasant Surprise
Sad
Dataset
Toronto Emotional Speech Set (TESS)

Dataset Source:

Available on Kaggle

The dataset contains:

Speech audio samples (.wav)
Corresponding transcript words
Emotion labels

Example filename:

OAF_back_angry.wav

Meaning:

back → transcript/text
angry → emotion label
Project Architecture
                +-------------------+
                |   TESS Dataset    |
                +-------------------+
                         |
          --------------------------------
          |                              |
          |                              |
+-------------------+      +----------------------+
| Speech Pipeline   |      | Text Pipeline        |
| MFCC + BiLSTM     |      | BERT Classifier      |
+-------------------+      +----------------------+
          |                              |
          ----------- Fusion -------------
                         |
              +-------------------+
              | Final Emotion     |
              | Prediction        |
              +-------------------+
Speech Pipeline
Steps
Audio preprocessing
MFCC feature extraction
Feature padding
Label encoding
BiLSTM model training
Emotion prediction
Technologies Used
PyTorch
Librosa
NumPy
Scikit-learn
Speech Model
BiLSTM Neural Network
Input Features: MFCC
Output: 7 emotion classes
Speech Results

Training Accuracy:

~99%

Speech emotion recognition produced strong performance because emotional information is highly present in vocal tone, pitch, and intensity.

Text Pipeline
Steps
Transcript extraction from TESS filenames
Text tokenization
BERT-based classification
Emotion prediction
Technologies Used
Transformers (HuggingFace)
PyTorch
BERT
Text Results

Training Accuracy:

~14%

The TESS transcripts consist mostly of semantically neutral single-word utterances, making standalone text emotion classification challenging.

Example transcript words:

back
bar
chair
calm

These words themselves do not strongly indicate emotions.

Fusion Pipeline
Purpose

The fusion pipeline combines:

Speech predictions
Text predictions

to produce a final multimodal emotion prediction.

Fusion Model
Fully Connected Neural Network
Speech Features + Text Features concatenated together
Output

Final predicted emotion among:

angry
disgust
fear
happy
neutral
pleasant_surprise
sad
Results Summary
Modality	Accuracy
Speech Pipeline	~99%
Text Pipeline	~14%
Fusion Pipeline	Working Successfully
Key Observation

Speech modality contributes significantly more emotional information compared to transcript text in the TESS dataset.

Technologies Used
Python
PyTorch
HuggingFace Transformers
Librosa
NumPy
Pandas
Scikit-learn
Folder Structure
Multimodal-Emotion-Recognition/
│
├── data/
│   └── raw/
│
├── models/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/
│
├── venv/
│
└── README.md
How to Run
1. Create Virtual Environment
python -m venv venv
2. Activate Environment
Windows
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
Run Speech Pipeline
Train Speech Model
python models/speech_pipeline/train.py
Test Speech Model
python models/speech_pipeline/test.py
Run Text Pipeline
Train Text Model
python models/text_pipeline/train.py
Test Text Model
python models/text_pipeline/test.py
Run Fusion Pipeline
Train Fusion Model
python models/fusion_pipeline/train.py
Test Fusion Model
python models/fusion_pipeline/test.py
Future Improvements
Real-time emotion recognition
Streamlit/Flask web application
Better multimodal fusion strategies
Larger transcript datasets
GPU-based transformer training
Attention-based fusion models
Conclusion

This project successfully implements a complete multimodal emotion recognition system using speech and text modalities.

The results demonstrate that:

Speech features are highly effective for emotion recognition.
Transcript text alone is comparatively weak for emotion classification in the TESS dataset.
Combining multiple modalities provides a complete multimodal AI system.