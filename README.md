# Stereotypes in pretrained language models

This Github repository contains the official code for the paper: *"Stepmothers are mean and academics are pretentious: What do pretrained language models learn about you?"*

## Getting started


<ul>
<li> Clone the repository: <code>git clone git@github.com:RochelleChoenni/stereotypes_in_lms.git </code>.</li>
<li> Install the required packages: <code>pip install  -r requirements.txt </code></li>
<li> Download the <a href="http://saifmohammad.com/WebDocs/Lexicons/NRC-Suite-of-Sentiment-Emotion-Lexicons.zip">NRC lexicon</a> and unzip in the root folder: <br>
  <code> wget http://saifmohammad.com/WebDocs/Lexicons/NRC-Suite-of-Sentiment-Emotion-Lexicons.zip && unzip NRC-Suite-of-Sentiment-Emotion-Lexicons.zip </code> <br>
  Move the lexicon into the right folder: <br>
 <code> mv NRC-Suite-of-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92-In105Language-Nov2017Translations.xlsx emotion_scores </code> <br>
  Delete unecessary data: <br>
  <code> rm -r NRC-Suite-of-Sentiment-Emotion-Lexicons && rm NRC-Suite-of-Sentiment-Emotion-Lexicons </code> 
</li>
<li> Download the <a href="https://www.kaggle.com/snapcrack/all-the-news">All-the-news dataset</a> (This should give you the file <code> archive.zip</code>) and extract: <br> 
  <code> unzip archive.zip -d news_data/archive </code>
  </li>
</ul>

## Collect data

Code for recreating the stereotypes dataset from autocomplete suggestions can be found in the folder <code>data_collection/</code>.\
From this folder first run e.g.: \
<code> python create_dataset.py --save_path stereo_dataset/stereotypes_data.csv </code> \
To clean up the data run e.g:\
<code> python clean_dataset.py --file_path stereo_dataset/stereotypes_data.csv --save_path stereo_dataset/cleaned_data.csv</code>\
*Note that the final dataset from the paper has been manually cleaned up as well.*

## Anaylzing results

This repo contains a **demo notebook** that contains an easy interface to further analyse the results from the paper. This notebook should be self-explanatory and instructions on how to use it are provided inside. 

## Running your own fine-tuning experiments
To run your own experiments first fine-tune a model on a data source of choice, e.g. using the code provided by the <a href="https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling">Hugging Face library</a>. Code for running experiments on probing the language model for stereotypes can be found in the file <code>probe_mlm.py</code>. To retrieve emotion profiles see <code>compute_emotion_scores.py</code>. 

## Citation
```bibtex
@inproceedings{choenni2021stepmothers,
    title={Stepmothers are mean and academics are pretentious: What do pretrained language models learn about you?},
    author={Choenni, Rochelle and Shutova, Ekaterina and van Rooij, Robert},
    booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2021}
}
```