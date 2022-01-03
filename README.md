# Learning with Different Amounts of Annotation: From Zero to Many Labels

This is the official GitHub repository for the following paper:

> [Learning with Different Amounts of Annotation: From Zero to Many Labels.](https://arxiv.org/abs/2109.04408)  
> Shujian Zhang, Chengyue Gong, and Eunsol Choi.  
> _EMNLP_, 2021.  


If using, please cite:
```
@article{zhang2021learning,
  title={Learning with different amounts of annotation: From zero to many labels},
  author={Zhang, Shujian and Gong, Chengyue and Choi, Eunsol},
  journal={arXiv preprint arXiv:2109.04408},
  year={2021}
}

```

### Pre-requisites

This repository is based on the [HuggingFace Transformers](https://github.com/huggingface/transformers) library.
<!-- Hyperparameter tuning is based on [HFTune](https://github.com/allenai/hftune). -->


### Setup
Install packages:

```
pip install -r requirements.txt
```


### Train GLUE-style model

To train a GLUE-style model using this repository:

```
TASK=snli
MODEL_OUTPUT_DIR=./model_output_snli_xxx/
Setting=xxx
Tra=xxx
Ft=xxx

python -m cartography.classification.run_glue \
    -c configs/$TASK.jsonnet \
    --do_train \
    --do_eval \
    --do_finetune \
    --num_train_epochs $Tra \
    --ft_num_train_epochs $Ft \
    --label_propagation \
    --setting $Setting \
    --overwrite_output_dir \
    --overwrite_cache \
    -o $MODEL_OUTPUT_DIR

```
The best configurations for our experiments for each of the `$TASK`s (SNLI and MNLI) are provided under [configs](./configs).

### Data Selection

To select (different amounts of) data based on various metrics from training dynamics:

```
python -m cartography.selection.train_dy_filtering \
    --filter \
    --task_name $TASK \
    --model_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS \
    --metric $METRIC \
    --data_dir $PATH_TO_GLUE_DIR_WITH_ORIGINAL_DATA_IN_TSV_FORMAT
```

Supported `$TASK`s include SNLI, QNLI, MNLI and WINOGRANDE, and `$METRIC`s include `confidence`, `variability`, `correctness`, `forgetfulness` and `threshold_closeness`; see [paper](https://arxiv.org/abs/2009.10795) for more details.

To select _hard-to-learn_ instances, set `$METRIC` as "confidence" and for _ambiguous_, set `$METRIC` as "variability". For _easy-to-learn_ instances: set `$METRIC` as "confidence" and use the flag `--worst`. 
