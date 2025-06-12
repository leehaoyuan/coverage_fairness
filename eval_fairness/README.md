# coverage_fairness
This repository presents the implementation of the NAACL 2025 paper:
> [**Coverage-based Fairness in Multi-document Summarization**](https://arxiv.org/pdf/2412.08795v2),<br/>
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
An aggregated news dataset ([link1](https://aclanthology.org/N16-1138.pdf),[link2](http://www.fakenewschallenge.org/),[link3](https://aclanthology.org/K19-1046.pdf)) with labels of stance on different claims. We preprocess the dataset by clustering the news with the same target.

The preprocessed dataset can be downloaded from the google drive [link](https://drive.google.com/file/d/1m8xdLAi7kkMQMrGAXS8O0JYSyI7135Jq/view?usp=sharing).

## Summary Generation
The generated summaries can be found in the folder `generated_summary`.

To generate summaries with different models on your own, please use the scripts in the folder `gen_summary`. Specifically, for generating summaries with Gemma2, Llama2, Llama3.1 and Mistral, please use `gen_summary/gen_summary_vllm.py`. For generating summaries with other models, please use other corresponding scripts. To generate summaries with COOP model, please install the coop package separately following the [instruction](https://github.com/megagonlabs/coop) since its requirements conflict with other packages. To use these scripts, please fill in the huggingface token, claude api key and openai api key in the appropriate place. Example usages of these scripts are shown below:

    python gen_summary/gen_summary_chatgpt.py --model gpt-3.5-turbo-0125 --input_dataset dataset/amazon300v3 --output_summary generated_summary/amazon300v3_summary_gpt3proc.json
  
    python gen_summary/gen_summary_vllm.py --model meta-llama/Llama-3.1-70B-Instruct --input_dataset dataset/semeval300v2 --num_gpu 4 --output_summary generated_summary/semeval_summary_llama3.170bproc.json
  
    python gen_summary/gen_summary_pegasus.py --model google/pegasus-cnn_dailymail --input_dataset dataset/mitweet300v2 --output_summary generated_summary/mitweet_summary_pegasusproc.json


## Fairness Evaluation

To evaluate the fairness of generated summaries, first use `gen_acu.py` to extract ACUs from the summaries. Please fill in the openai api key in the appropriate place. The example usage of `gen_acu.py` is shown below:

	python gen_acu.py --input generated_summary/semeval_summary_llama3.170bproc.json --dataset semeval
The above line generates a file containing extracted ACUs of the summaries named `semeval_summary_llama3.170bproc_acu_preprocessed.json`. Then, use `eval_fairness.py` to evaluate the fairness given the extracted ACUs. The example usage of `eval_fairness.py` is shown below:
	
 	python eval_fairness.py --model roberta-large-mnli --input_dataset semeval300v2 --input_acu semeval300v2_summary_llama3.170bproc_acu_preprocessed
The above line outputs the Equal Coverage and Coverage Parity scores as well as overrepresented and underrepreseted social attribute values. The example output is as follows:

```
Equal Coverage: 0.0302
Coverage Parity: 0.003
Coverage Probability Difference: support: 0.0045 against: -0.0044
Most overrepresented social attribute value: support significance: 0.01
Most underrepresented social attribute value: against significance: 0.004
```
