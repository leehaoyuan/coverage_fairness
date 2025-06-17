# FairPO
This repository presents the implementation of the ACL 2025 paper:
> [**Improving Fairness of Large Language Models in Multi-document Summarization**](https://arxiv.org/pdf/2506.07479),<br/>
[Haoyuan Li](https://leehaoyuan.github.io/), [Rui Zhang](https://ryanzhumich.github.io/), and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/)

## Data
The training, validation, and test sets of each dataset can be downloaded from ([link](https://drive.google.com/file/d/1ygf-7W4N9zOpmLhrBNNlKmG7tqrQnu2c/view?usp=drive_link)). For each dataset, we provide three different splittings of training, validation, and test, denoted as `batch1`, `batch2`, and `batch3`.

## Perturbation-based Preference Pair Generation
To generate preference pairs, FairPO first generate a candidate summary based on the full input document set using the following command line:
```
python gen_preference_candidate.py --input_file final_data/amazon_batch1_train --input_model meta-llama/Llama-3.1-8B-Instruct --seed 0
```
The above command line will generate summaries and store them in a file named `amazon_batch1_train_llama3proc0.json`. Then, please use the codes under `eval_fairness` to caclulate the summary-level fairness score for each generated summary. 

The processed file for preference tuning can be downloaded from ([link](https://drive.google.com/file/d/1S5T0FF_xFnq4Jt6t42jon3v5LteFc3gS/view?usp=sharing)).
