# FairPO
This repository presents the implementation of the ACL 2025 paper:
> [**Improving Fairness of Large Language Models in Multi-document Summarization**](https://arxiv.org/pdf/2506.07479),<br/>
[Haoyuan Li](https://leehaoyuan.github.io/), [Rui Zhang](https://ryanzhumich.github.io/), and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/)

## Data
The training, validation, and test sets of each dataset can be downloaded from ([link](https://drive.google.com/file/d/1ygf-7W4N9zOpmLhrBNNlKmG7tqrQnu2c/view?usp=drive_link)). After downloading the file, please unzip it under the current (`FairPO`) directory. For each dataset, we provide three different splittings of training, validation, and test, denoted as `batch1`, `batch2`, and `batch3`.

## Perturbation-based Preference Pair Generation
To generate preference pairs, FairPO first generate a candidate summary based on the full input document set using the following command line:
```
python gen_preference_candidate.py --input_file final_data/amazon_batch1_train --input_model meta-llama/Llama-3.1-8B-Instruct --seed 0
```
The above command line will generate summaries and store them in a file named `amazon_batch1_train_llama3proc0.json`. Then, please use the codes under `eval_fairness` to caclulate the summary-level fairness score for each generated summary.

```
cd ..
python eval_fairness/gen_acu.py --input amazon_batch1_train_llama3proc0 --dataset amazon
python eval_fairness/eval_fairness.py --model roberta-large-mnli --input_dataset FairPO/final_data/amazon_batch1_train --input_acu amazon_batch1_train_llama3proc0_acu_preprocessed
cp amazon_batch1_train_llama3proc0_ecv2.pickle eval_fairness/.
```

The above lines generate a file named `amazon_batch1_train_llama3proc0_ecv2.pickle`, which stores the equal coverage value for each summary. Then, use `perturbation.py` to generate perturbed input document sets.

```
python peturbation.py --input_dataset final_data/amazon_batch1_train --input_ec_file amazon_batch1_train_llama3proc0_ecv2
```

The above line generate two datasets storing perturbed input documents: `final_data/amazon_batch1_train_middle.pickle` and `final_data/amazon_batch1_train_order.pickle`. Please refer to the above lines to generate summaries for these two datasets and evaluate fairness for the generated summaries.

```
python gen_preference_candidate.py --input_file final_data/amazon_batch1_train_middle --input_model meta-llama/Llama-3.1-8B-Instruct --seed 0
python gen_preference_candidate.py --input_file final_data/amazon_batch1_train_order --input_model meta-llama/Llama-3.1-8B-Instruct --seed 0
cd ..
python eval_fairness/gen_acu.py --input amazon_batch1_train_middle_llama3proc0 --dataset amazon
python eval_fairness/eval_fairness.py --model roberta-large-mnli --input_dataset FairPO/final_data/amazon_batch1_train --input_acu amazon_batch1_train_middle_llama3proc0_acu_preprocessed
cp amazon_batch1_train_middle_llama3proc0_ecv2.pickle eval_fairness/.
python eval_fairness/gen_acu.py --input amazon_batch1_train_order_llama3proc0 --dataset amazon
python eval_fairness/eval_fairness.py --model roberta-large-mnli --input_dataset FairPO/final_data/amazon_batch1_train --input_acu amazon_batch1_train_order_llama3proc0_acu_preprocessed
cp amazon_batch1_train_order_llama3proc0_ecv2.pickle eval_fairness/.
```

Eventually, please use the `gen_preference_data.py` to generate preference pairs for preference tuning.
```
python gen_preference_data.py --model meta-llama/Llama-3.1-8B-Instruct --summary1 amazon_batch1_train_llama3proc0 --summary2 amazon_batch1_train_middle_llama3proc0 --summary3 amazon_batch1_train_order_llama3proc0 --data final_data/amazon_batch1_train --output_path fairpo_summary_llama3_amazon_batch1_train
```
The above line generates a file named `fairpo_summary_llama3_amazon_batch1_train.pickle` which stores the generated preference pairs. The processed files for preference tuning can be downloaded from ([link](https://drive.google.com/file/d/1S5T0FF_xFnq4Jt6t42jon3v5LteFc3gS/view?usp=sharing)).

## Fairness-aware Preference Tuning
To perform fairness-aware preference tuning on the generated preference pairs, please use the following command lines:
```
python fairpo_train.py \
  --train_data fairpo_summary_llama3_amazon_batch1_train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output_path fairpo_summary_llama3_amazon_batch1 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_bias_label 3 \
  --weight_schema fairpo \
  --weight_step 0.75 \
  --temperature 1.0
```

