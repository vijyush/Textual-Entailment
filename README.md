# Textual-Entailment
Contradiction? Let me check.
For any two sentences, there are three ways they could be related: one could entail the other, one could contradict the other, or they could be unrelated. Natural Language Inferencing (NLI) is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) are related.Your task is to create an NLI model that assigns labels of 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses.

## DESCRIPTION:
This repository contains the implementation of a Textual Entailment project using the Stanford Natural Language Inference (SNLI) dataset.The main objective of this project is to develop and evaluate models that can accurately determine whether a given hypothesis is entailed by a premise. The models are trained on the SNLI dataset, which provides labeled sentence pairs.The project was developed and executed using Google Colab, leveraging its powerful computational resources for deep learning tasks.

### Requirements:
[Download the SNLI datasets from the official website:](https://nlp.stanford.edu/projects/snli/)
Necessary libraries and modules to run the code:
-Pandas
-Matplotlib
-Seaborn
-Numpy
-Natural Language Toolkit(nltk)
-contractions
-Regular Expression(re)
-Tensorflow
-tf-keras
-Transformers

### Project Approach:
#### 1.Loading the datasets:
-Download the datasets from above link and save to the google drive.
-Mount your google drive to access the datasets and define the path to your dataset files located in google drive(Replace your path in the code to run successfully).
#### 2.Expolaratory Data Analysis:
- summary of the dataset including shape, data types, and basic statistics.
- Identify any missing values,duplicates values and eliminated .
- Analysing the distribution of entailment labels(i.e.,gold_labels).
#### 3.Text Analysis:
 -Calculation and visualisation of the length of hypothesis(sentence_1) and premise(sentence_2) texts.
#### 4.Overview of Word Distribution:
 -Understanding the distribution of words in the dataset(count of the words & frequency of words).
 -Using wordcloud for visualisation of frequency of words.
#### 5.Tokenization:
-Tokenization: Breaks down text into subword tokens.
-Encoding: Converts tokens into numerical IDs with padding and truncation.
-Converting to Tensors: Prepares the encoded input for model processing.
-creating TensorFlow datasets to handle the data pipeline efficiently.
#### 6.Model Setup:
-Load and set up the DistilBERT model for sequence classification.
-Compile the model with appropriate loss and optimizer.
-Evaluate the model on the test set.
#### Summary:
-Setup and Install: Install necessary libraries.
-Prepare Data: Load and preprocess data.
-Build and Train: Set up and train the model.
-Evaluate: Evaluate the model’s performance.
### DistillBERT 
        DistilBERT is a smaller, faster, and lighter version of BERT, designed to be more efficient while retaining a significant portion of BERT's performance.DistilBERT has approximately 60% of the parameters of the original BERT model and retains about 97% of BERT's performance on various NLP benchmarks.It is faster to train and infer due to its reduced size, which is advantageous for large-scale datasets like SNLI Datasets(used in above project).It captures complex contextual relationships in text through self-attention mechanisms.It leverages pre-trained knowledge to perform well on textual entailment tasks after fine-tuning on specific datasets.

### Resources:
![SNLI DATASETS](https://nlp.stanford.edu/projects/snli/)
![Reference for EDA](https://medium.com/@navamisunil174/exploratory-data-analysis-of-breast-cancer-survival-prediction-dataset-c423e4137e38)
![Reference to know the structure of the project](https://www.kaggle.com/code/nupurroy/kernel674ca09f2c#XLM-RoBERTa-Model:-(Large))
![Reference for Wordcloud](https://medium.com/@natashanewbold/creating-a-wordcloud-using-python-a905efc3c288)

###### Drawback: The accuracy is not as expected.we have to work on it.




 




   
