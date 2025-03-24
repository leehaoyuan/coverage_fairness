# coverage_fairness
This repository presents the implementation of the NAACL 2024 paper:
> [** Coverage-based Fairness in Multi-document Summarization**](https://arxiv.org/pdf/2412.08795),<br/>
[Haoyuan Li](https://leehaoyuan.github.io/), [Yusen Zhang](https://yuszh.com/), [Rui Zhang](https://ryanzhumich.github.io/), and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/)

## Data
We preprocess the following list of datasets and convert them into multi-document summarization datasets with labels of social attributes:
### Amazon
A review dataset ([link](https://nijianmo.github.io/amazon/index.html)) with labels of sentiment (converted from rating). We preprocess the dataset by sampling from it.

### MITweet 
A tweet dataset ([link](https://github.com/LST1836/MITweet) with labels of political ideologies. We preprocess the dataset by clustering the tweets of the same topic.

### SemEval
A tweet dataset ([link](https://huggingface.co/datasets/krishnagarg09/SemEval2016Task6) with labels of stance on different targets. We preprocess the dataset by clustering the tweets with the same target.

### Article Bias
A news dataset ([link](https://github.com/ramybaly/Article-Bias-Prediction)) with labels of political ideologies. We preprocess the dataset by clustering the news based on dates and tfidf similarity

### News Stance
An aggregated news dataset ((link1)[https://aclanthology.org/N16-1138.pdf],[link2](http://www.fakenewschallenge.org/),[link3](https://aclanthology.org/K19-1046.pdf)) with labels of stance on different claims. We preprocess the dataset by clustering the news with the same target.

The preprocessed dataset can be downloaded from the google drive [link](https://drive.google.com/file/d/1m8xdLAi7kkMQMrGAXS8O0JYSyI7135Jq/view?usp=sharing)

## Environment
