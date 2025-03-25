# coverage_fairness
This repository presents the implementation of the NAACL 2025 paper:
> [**Coverage-based Fairness in Multi-document Summarization**](https://arxiv.org/pdf/2412.08795),<br/>
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

The preprocessed dataset can be downloaded from the google drive [link](https://drive.google.com/file/d/1m8xdLAi7kkMQMrGAXS8O0JYSyI7135Jq/view?usp=sharing).

## Environment
The environment for this project can be created based on the following intrcustions:

* __Python version:__ `python3.10`

* __Dependencies:__ Use the `requirements.txt` file and conda/pip to install all necessary dependencies. E.g., for pip:

		pip install -U pip
		pip install -U setuptools
		pip install -r requirements.txt 

## Summary Generation
To generate summaries with different models, please use the scripts in the folder `gen_summary`. Specifically, for generating summaries with Gemma2, Llama2, Llama3.1 and Mistral, please use `gen_summary/gen_summary_vllm.py`. For generating summaries with other models, please use other corresponding scripts. To generate the summary with COOP model, please install the coop package separately following the [instruction](https://github.com/megagonlabs/coop) since its requirements conflict with other packages. To use these scripts, please fill in the huggingface token, claude api key and openai api key in the appropriate place. Example usage of these scripts are shown below:

    python gen_summary/gen_summary_chatgpt.py --model gpt-3.5-turbo-0125 --input_dataset dataset/amazon300v3 --output_summary generated_summary/amazon300v3_summary_gpt3proc.json
  
    python gen_summary/gen_summary_vllm.py --model meta-llama/Llama-3.1-70B-Instruct --input_dataset dataset/semeval300v2 --num_gpu 4 --output_summary generated_summary/semeval_summary_llama3.170bproc.json
  
    python gen_summary/gen_summary_pegasus.py --model google/pegasus-cnn_dailymail --input_dataset dataset/mitweet300v2 --output_summary generated_summary/mitweet_summary_pegasusproc.json

The generated summaries can be found in the folder `generated_summary`.

## Fairness Evaluation
