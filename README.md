# Stereotypes in pretrained language models

This Github repository contains the official code for the paper "Stepmothers are mean and academics are pretentious: What do pretrained language models learn about you?"

## Getting started


<ul>
<li> Clone the repository: <code>git clone git@github.com:RochelleChoenni/stereotypes_in_lms.git </code>.</li>
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
First run: \
<code> python data_collection/create_dataset.py </code> \
To clean up the data run:\
<code> python data_collection/clean_dataset.py</code>\
*Note that the final dataset from the paper has been manually cleaned up as well.*


