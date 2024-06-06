<h1 align="center">Multi-Objective Linguistic Control of<br>Large Language Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2406.16229">📃 Paper</a> •
  <a href="https://huggingface.co/umd-zhou-lab/controllable-llama2-7b">🤗 Model (controllable-llama2-7b)</a> •
  <a href="https://huggingface.co/umd-zhou-lab/controllable-wizardlm-7b">🤗 Model (controllable-wizardlm-7b)</a>
</p>

![Overview](assets/cover.webp)

## Installation

To install the necessary dependencies, use the following commands:

```bash
conda create -n mctune python=3.9
conda activate mctune
pip install -r requirements.txt
```

## Data Preparation for MCTune

To prepare data for multi-control tuning, use the following commands. We support Alpaca-GPT4 and a subset of the Alpaca-evol-instruct-70k (first 50k examples) natively. Any other instruction-tuning datasets in the same format as Alpaca can also be used.

The `--scale` parameter adjusts the difficulty of the test set. A larger scale results in more diverse control vectors further from the training distribution.

```bash
# Prepare train and test sets for MCTune from Alpaca-GPT4
python prepare_data.py --input_data_path ./data/alpaca_gpt4_data.json --scale 0.1

# Prepare train and test sets for MCTune from Alpaca-evol-instruct-50k
python prepare_data.py --input_data_path ./data/alpaca_evol_instruct_50k.json --scale 0.1
```

## Training

After creating the train and test sets, start MCTune with the following command:

```bash
torchrun --nproc_per_node=4 --master_port=8889 train.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --train_data_path ./data/tagged_alpaca_gpt4_data_quant_none_N_train_1_N_test_5_max_num_tags_5_scale_0.1_new_eval_train.json \
    --bf16 True \
    --output_dir ./models/controllable-llama2-7b \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```

## Evaluation

To evaluate the model, run the following command. Multi-GPU evaluation is supported for faster inference:

```bash
accelerate launch \
    --config_file ./accelerate_config.json \
    eval.py \
    --model_name_or_path ./models/controllable-llama2-7b \
    --data_path ./data/tagged_alpaca_gpt4_data_quant_none_N_train_1_N_test_5_max_num_tags_5_scale_0.1_new_eval_test.json \
    --output_dir . \
    --eval_batch_size 8 \
    --model_max_length 1024 \
    --output_path ./results/controllable-llama2-7b-alpaca-gpt4-eval_results-10.json
```

## Citation

```bibtex
@misc{nguyen2024multiobjectivelinguisticcontrollarge,
  title={Multi-Objective Linguistic Control of Large Language Models}, 
  author={Dang Nguyen and Jiuhai Chen and Tianyi Zhou},
  year={2024},
  eprint={2406.16229},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2406.16229}, 
}
```