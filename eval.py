#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
from tqdm import tqdm
import lftk
import spacy
import pdb
import sys
from accelerate import Accelerator
import accelerate
import torch.distributed as dist
import random
import time
import numpy as np
import os
from copy import deepcopy

from utils import get_lftk_mappings


def gather_object(object):
    output_objects = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(output_objects, object)
    return [x for y in output_objects for x in y]


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "{meta_instruction}\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "{meta_instruction}\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_input_few_shot": (
        "{meta_instruction}\n\n"
        "{demos}\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input_few_shot": (
        "{meta_instruction}\n\n"
        "{demos}\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./models/controllable-llama2-7b")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the eval data."})
    num_samples: Optional[int] = field(default=None, metadata={"help": "Number of subsamples to evaluate on. Default is None, which means evaluate on all samples."})
    response_cache: Optional[str] = field(default=None)
    few_shot: bool = field(default=False)
    output_path: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    eval_batch_size: int = field(default=8, metadata={"help": "Batch size for evaluation."})
    seed: int = field(default=0, metadata={"help": "Random seed."})
    use_lora: bool = field(default=False)
    temperature: float = field(default=1.0)
    do_sample: bool = field(default=False)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, few_shot: bool=False):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        if isinstance(list_data_dict, dict):
            if "metadata" in list_data_dict:
                self.metadata = list_data_dict["metadata"]
                self.lftk_ranges = self.metadata["feat_ranges"]
                list_data_dict = list_data_dict["data"]
            else:
                raise ValueError(f"Invalid data format. Please check the data format of {data_path}.")

        self.list_data_dict = list_data_dict

        logging.warning("Formatting inputs...")
        if not few_shot:
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        else:
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_few_shot"], PROMPT_DICT["prompt_no_input_few_shot"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        self.inputs = sources

        lftk_key2id, _ = get_lftk_mappings()

        self.num_tags = [torch.as_tensor(example["num_tags"]) for example in list_data_dict]
        self.tag_ids = [torch.as_tensor([lftk_key2id[tag] for tag in example["selected_tags"]]) for example in list_data_dict]
        self.tag_values = [torch.as_tensor(example["tag_values"]) for example in list_data_dict]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(example_id=i, inputs=self.inputs[i], num_tags=self.num_tags[i], tag_ids=self.tag_ids[i], tag_values=self.tag_values[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        example_ids = torch.as_tensor([instance["example_id"] for instance in instances])
        inputs = [instance["inputs"] for instance in instances]
        inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask

        num_tags, tag_ids, tag_values = tuple([instance[key] for instance in instances] for key in ("num_tags", "tag_ids", "tag_values"))
        num_tags = torch.as_tensor(num_tags)
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=IGNORE_INDEX)
        tag_values = torch.nn.utils.rnn.pad_sequence(tag_values, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            example_ids=example_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_tags=num_tags,
            tag_ids=tag_ids,
            tag_values=tag_values,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for evaluation."""
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, few_shot=data_args.few_shot)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=None, eval_dataset=eval_dataset, data_collator=data_collator)


def eval():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map={"": accelerator.process_index},
    )
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    eval_dataset = data_module["eval_dataset"]
    if data_args.num_samples is not None:
        subset_indices = range(data_args.num_samples)
        eval_subset = torch.utils.data.Subset(eval_dataset, subset_indices)
    else:
        subset_indices = None
        eval_subset = eval_dataset
    print(f"len(eval_subset)={len(eval_subset)}")

    accelerator.wait_for_everyone()

    start = time.time()
    with accelerator.split_between_processes(list(range(len(eval_subset)))) as subsubset_indices:
        eval_subsubset = torch.utils.data.Subset(eval_subset, subsubset_indices)
        eval_subsubsetloader = DataLoader(
            eval_subsubset,
            batch_size=training_args.eval_batch_size,
            collate_fn=data_module["data_collator"],
            pin_memory=True,
        )

        total_tags = 0
        total_correct_tags = 0
        num_samples = len(eval_subsubset)
        num_strict_follow = 0
        num_loose_follow = 0
        count_per_tag = {feat: 0 for feat in eval_dataset.metadata["selected_features"]}
        correct_per_tag = {feat: 0 for feat in eval_dataset.metadata["selected_features"]}
        l1_error_per_tag = {feat: 0 for feat in eval_dataset.metadata["selected_features"]}
        l2_error_per_tag = {feat: 0 for feat in eval_dataset.metadata["selected_features"]}
        per_example_results = []

        nlp = spacy.load("en_core_web_sm")
        lftk_key2id, lftk_id2key = get_lftk_mappings()

        for batch in tqdm(eval_subsubsetloader, disable=(not accelerator.is_local_main_process)):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            example_ids = batch["example_ids"].tolist()
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            num_tags, tag_ids, tag_values = batch["num_tags"], batch["tag_ids"], batch["tag_values"]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=tokenizer.model_max_length,
                temperature=training_args.temperature,
                do_sample=training_args.do_sample,
            )
            decoded_outputs = tokenizer.batch_decode(outputs[:, input_ids.shape[-1]:], skip_special_tokens=True)

            for idx, decoded_output in enumerate(decoded_outputs):
                doc = nlp(decoded_output)
                selected_tag_ids = tag_ids[idx][:num_tags[idx]]
                selected_tags = [lftk_id2key[tag_id] for tag_id in selected_tag_ids.tolist()]
                extracted_features = lftk.Extractor(docs=doc).extract(selected_tags)
                assert list(extracted_features.keys()) == selected_tags, "extracted_features are not in the same order as selected_tags"

                tag_value_preds = torch.as_tensor(list(extracted_features.values()), device=accelerator.device)

                matches = (tag_value_preds == tag_values[idx][:num_tags[idx]]).int()
                l1_error = torch.abs(tag_value_preds - tag_values[idx][:num_tags[idx]])
                l2_error = (tag_value_preds - tag_values[idx][:num_tags[idx]]) ** 2

                per_example_result = {
                    "example_idx": example_ids[idx],
                    "response": decoded_output,
                    "predicted_tag_values": list(extracted_features.values()),
                    "matches": matches.tolist(),
                    "l1_error": l1_error.tolist(),
                    "l2_error": l2_error.tolist(),
                }
                per_example_results.append(per_example_result)

                # Compute metrics
                # To compute zero_one_score
                total_tags += num_tags[idx].item()
                total_correct_tags += matches.sum().item()

                # To compute rate of strict/loose follows
                if matches.sum() == num_tags[idx]:
                    num_strict_follow += 1
                if matches.sum() > 0:
                    num_loose_follow += 1

                # To compute accuracy per tag
                for i, tag in enumerate(selected_tags):
                    count_per_tag[tag] += 1
                    correct_per_tag[tag] += matches[i].item()
                    l1_error_per_tag[tag] += l1_error[i].item()
                    l2_error_per_tag[tag] += l2_error[i].item()

        results = {
            "subsubset_indices": subsubset_indices,
            "total_tags": total_tags,
            "total_correct_tags": total_correct_tags,
            "num_strict_follow": num_strict_follow,
            "num_loose_follow": num_loose_follow,
            "count_per_tag": count_per_tag,
            "correct_per_tag": correct_per_tag,
            "l1_error_per_tag": l1_error_per_tag,
            "l2_error_per_tag": l2_error_per_tag,
            "per_example_results": per_example_results,
        }
        results = [results]

    if accelerator.num_processes > 1:
        results = accelerator.gather_for_metrics(results)

    if accelerator.is_main_process:
        stop = time.time()
        print(f"Number of examples evaluated: {len(eval_subset)}")
        print(f"Time taken: {stop - start:.2f}s")
        print(f"Time per example: {(stop - start) / len(eval_subset):.2f}s")

        # Aggregate results
        total_tags = 0
        total_correct_tags = 0
        num_samples = len(eval_subset)
        num_strict_follow = 0
        num_loose_follow = 0
        count_per_tag = {feat: 0 for feat in eval_dataset.metadata["selected_features"]}
        correct_per_tag = {feat: 0 for feat in eval_dataset.metadata["selected_features"]}
        l1_error_per_tag = {feat: 0 for feat in eval_dataset.metadata["selected_features"]}
        l2_error_per_tag = {feat: 0 for feat in eval_dataset.metadata["selected_features"]}
        for result in results:
            total_tags += result["total_tags"]
            total_correct_tags += result["total_correct_tags"]
            num_strict_follow += result["num_strict_follow"]
            num_loose_follow += result["num_loose_follow"]
            for tag, count in result["count_per_tag"].items():
                count_per_tag[tag] += count
                correct_per_tag[tag] += result["correct_per_tag"][tag]
                l1_error_per_tag[tag] += result["l1_error_per_tag"][tag]
                l2_error_per_tag[tag] += result["l2_error_per_tag"][tag]
        per_example_results = sum([result["per_example_results"] for result in results], [])

        # Save fine-grained results
        list_data_dict = eval_dataset.list_data_dict
        for res in per_example_results:
            list_data_dict[res["example_idx"]].update(res)

        # Compute metrics
        zero_one_score = total_correct_tags / total_tags
        strictly_followed = num_strict_follow / num_samples
        loosely_followed = num_loose_follow / num_samples
        accuracy_per_tag = {tag: ((correct_per_tag[tag] / count_per_tag[tag]) if count_per_tag[tag] > 0 else np.nan) for tag in eval_dataset.metadata["selected_features"]}
        mae_per_tag = {tag: ((l1_error_per_tag[tag] / count_per_tag[tag]) if count_per_tag[tag] > 0 else np.nan) for tag in eval_dataset.metadata["selected_features"]}
        mse_per_tag = {tag: ((l2_error_per_tag[tag] / count_per_tag[tag]) if count_per_tag[tag] > 0 else np.nan) for tag in eval_dataset.metadata["selected_features"]}

        results = {
            "per_example_results": list_data_dict,
            "zero_one_score": zero_one_score,
            "strictly_followed": strictly_followed,
            "loosely_followed": loosely_followed,
            "accuracy_per_tag": accuracy_per_tag,
            "mae_per_tag": mae_per_tag,
            "mse_per_tag": mse_per_tag,
        }

        print(f"zero_one_score: {zero_one_score:.4f}")
        print(f"strictly_followed: {strictly_followed:.4f}")
        print(f"loosely_followed: {loosely_followed:.4f}")
        print("accuracy_per_tag:")
        for tag, accuracy in accuracy_per_tag.items():
            print(f"\t{tag}: {accuracy:.4f}")
        print("mae_per_tag:")
        for tag, mae in mae_per_tag.items():
            print(f"\t{tag}: {mae:.4f}")
        print("mse_per_tag:")
        for tag, mse in mse_per_tag.items():
            print(f"\t{tag}: {mse:.4f}")

        utils.jdump(results, f"{data_args.output_path}")
        print(f"Results saved to {data_args.output_path}")


if __name__ == "__main__":
    eval()
