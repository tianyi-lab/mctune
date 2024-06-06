import argparse
import concurrent.futures
import json
import os
import random
from typing import List

import lftk
import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm
from lftk.utils import safe_division
from utils import make_tag_description

nlp = spacy.load("en_core_web_sm")
lftk_feature_map = lftk.utils.get_feature_map(lftk.lftk.FEATURE_MAP_PATH)


class Quantizer:
    def __init__(self, ranges: dict[str, tuple], float_feats: set, quantization_option: str = "1to10"):
        self.ranges = ranges
        self.float_feats = float_feats
        self.quantization_option = quantization_option

    def do(self, tags: dict):
        if self.quantization_option == "none":
            return tags

        res = {}
        for tag, tag_value in tags.items():
            if tag in self.float_feats:
                if self.quantization_option == "1to5":
                    res[tag] = (tag_value - self.ranges[tag]["min"]) / (self.ranges[tag]["max"] - self.ranges[tag]["min"])
                    res[tag] = round(res[tag] * 4 + 1)
                elif self.quantization_option == "1to10":
                    res[tag] = (tag_value - self.ranges[tag]["min"]) / (self.ranges[tag]["max"] - self.ranges[tag]["min"])
                    res[tag] = round(res[tag] * 9 + 1)
        return res

    def undo(self, tags: dict):
        if self.quantization_option == "none":
            return tags

        res = {}
        for tag, tag_value in tags.items():
            if tag in self.float_feats:
                if self.quantization_option == "1to5":
                    res[tag] = (tag_value - 1) / 4
                    res[tag] = res[tag] * (self.ranges[tag]["max"] - self.ranges[tag]["min"]) + self.ranges[tag]["min"]
                elif self.quantization_option == "1to10":
                    res[tag] = (tag_value - 1) / 9
                    res[tag] = res[tag] * (self.ranges[tag]["max"] - self.ranges[tag]["min"]) + self.ranges[tag]["min"]
        return res


class Standardizer:
    def __init__(self, means: dict[str, float], sigmas: dict[str, float]):
        self.means = means
        self.sigmas = sigmas

    def do(self, tags: dict):
        res = {}
        for tag, tag_value in tags.items():
            res[tag] = (tag_value - self.means[tag]) / self.sigmas[tag]
        return res

    def undo(self, tags: dict):
        res = {}
        for tag, tag_value in tags.items():
            res[tag] = tag_value * self.sigmas[tag] + self.means[tag]
        return res


class TagSampler:
    def __init__(self, quantizer: Quantizer, standardizer: Standardizer, scale=0.1):
        self.quantizer = quantizer
        self.standardizer = standardizer
        self.scale = scale
        self.quantization_option = self.quantizer.quantization_option

    def is_valid(self, tags: dict):
        # Given a tag vector, return if it is valid
        # Note that the tag values should be quantized before calling this function 

        if "simp_ttr" in tags or "simp_noun_var" in tags or "simp_verb_var" in tags or "simp_adj_var" in tags or "rt_average" in tags:
            raise ValueError("tags should not contain derived features")

        if tags["t_word"] <= 0:
            return False
        if tags["t_word"] < (tags["n_noun"] + tags["n_verb"] + tags["n_adj"]):
            return False
        if tags["t_word"] < tags["t_uword"]:
            return False

        if tags["n_noun"] < 0:
            return False
        if tags["n_noun"] < tags["n_unoun"]:
            return False

        if tags["n_verb"] < 0:
            return False
        if tags["n_verb"] < tags["n_uverb"]:
            return False

        if tags["n_adj"] < 0:
            return False
        if tags["n_adj"] < tags["n_uadj"]:
            return False

        if tags["t_uword"] <= 0:
            return False
        if tags["t_uword"] < (tags["n_unoun"] + tags["n_uverb"] + tags["n_uadj"]):
            return False

        if tags["n_unoun"] < 0 or tags["n_uverb"] < 0 or tags["n_uadj"] < 0:
            return False

        if self.quantization_option == "1to5":
            if tags["fkre"] < 1 or tags["fkre"] > 5:
                return False
        elif self.quantization_option == "1to10":
            if tags["fkre"] < 1 or tags["fkre"] > 10:
                return False
        elif self.quantization_option == "none":
            if tags["fkre"] > 121.22:
                return False

        return True

    def update_tag(self, tags: dict, delta: torch.Tensor):
        res = {}
        for i, (tag, tag_value) in enumerate(tags.items()):
            res[tag] = tag_value + delta[i].item()
        return res

    def _sample(self, tags: dict):
        delta = torch.randn(len(tags)) * self.scale
        standardized_tags = self.standardizer.do(tags)
        sampled_tags = self.update_tag(standardized_tags, delta)
        sampled_tags = self.standardizer.undo(sampled_tags)
        return sampled_tags

    def sample(self, tags: dict, max_trials=10000) -> tuple[dict, bool]:
        # Given a tag vector x, we want to sample a tag vector y from the standard normal distribution centered around x
        # Only sample foundational features. Derived features are derived from foundational features.

        # Remove derived features
        tags_ = {k: v for k, v in tags.items() if k not in ["simp_ttr", "simp_noun_var", "simp_verb_var", "simp_adj_var", "rt_average"]}

        # Sample until we get a valid tag vector
        num_trials = 1
        sampled_tags = self._sample(tags_)
        while not self.is_valid(sampled_tags) and num_trials < max_trials:
            sampled_tags = self._sample(tags_)
            num_trials += 1

        if not self.is_valid(sampled_tags):
            return tags, False

        # Add the derived features back
        sampled_tags["simp_ttr"] = safe_division(sampled_tags["t_uword"], sampled_tags["t_word"])
        sampled_tags["simp_noun_var"] = safe_division(sampled_tags["n_unoun"], sampled_tags["n_noun"])
        sampled_tags["simp_verb_var"] = safe_division(sampled_tags["n_uverb"], sampled_tags["n_verb"])
        sampled_tags["simp_adj_var"] = safe_division(sampled_tags["n_uadj"], sampled_tags["n_adj"])
        sampled_tags["rt_average"] = safe_division(sampled_tags["t_word"], 240)

        sampled_tags["simp_ttr"] = self.quantizer.do({"simp_ttr": sampled_tags["simp_ttr"]})["simp_ttr"]
        sampled_tags["simp_noun_var"] = self.quantizer.do({"simp_noun_var": sampled_tags["simp_noun_var"]})["simp_noun_var"]
        sampled_tags["simp_verb_var"] = self.quantizer.do({"simp_verb_var": sampled_tags["simp_verb_var"]})["simp_verb_var"]
        sampled_tags["simp_adj_var"] = self.quantizer.do({"simp_adj_var": sampled_tags["simp_adj_var"]})["simp_adj_var"]
        sampled_tags["rt_average"] = self.quantizer.do({"rt_average": sampled_tags["rt_average"]})["rt_average"]

        # Round everything (if needed)
        for tag, tag_value in sampled_tags.items():
            if self.quantization_option in ["1to5", "1to10"]:
                sampled_tags[tag] = round(tag_value)
            elif self.quantization_option == "none":
                sampled_tags[tag] = round(tag_value, ndigits=2) if tag in self.quantizer.float_feats else round(tag_value)

        return sampled_tags, True


class Augmenter:
    def __init__(self,
        selected_features,
        quantization_option,
        num_aug=1,
        max_num_tags=-1,
        same_tag_per_aug=False,
        sample_tag_values=False,
        tag_sampler=None,
        few_shot=False,
        augment=False
    ):
        """
        num_aug: number of augmented examples per example
        max_num_tags: maximum number of tags to sample
        same_tag_per_aug: whether to sample the same tags for all augmented examples
        sample_tag_values: whether to sample new tag values or not
        few_shot: whether to generate a few-shot dataset or not
        augment: whether to perform augmentation or not
        """
        self.selected_features = selected_features
        self.quantization_option = quantization_option

        self.num_aug = num_aug
        self.max_num_tags = max_num_tags
        self.same_tag_per_aug = same_tag_per_aug
        self.sample_tag_values = sample_tag_values
        self.tag_sampler = tag_sampler
        self.few_shot = few_shot
        self.do_augment = augment 

        if not augment:
            self.meta_instruction = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request."
            )
        else:
            self.meta_instruction = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "At the end of the input, there will be a list of tags specifying the desired properties of the response. "
                f"The following tags are available: {make_tag_description(selected_features, quantization_option)} "
                "Write a response that appropriately completes the request and satisfies the tags."
            )

    def augment_row_once(self, row: dict, selected_tags: List[str]) -> dict:
        if not self.do_augment:
            return {
                "meta_instruction": self.meta_instruction,
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
            }

        groundtruth_tags = {k: v for k, v in row.items() if k in self.selected_features}
        if self.sample_tag_values:
            sampled_tags, _ = self.tag_sampler.sample(groundtruth_tags)
            tag_values = [sampled_tags[tag] for tag in selected_tags]
        else:
            tag_values = [(round(row[tag], 2) if round(row[tag]) != row[tag] else int(row[tag])) for tag in selected_tags]

        formatted_tags = " ".join([f"[{tag}: {tag_value}]" for tag, tag_value in zip(selected_tags, tag_values)])
        separator = " " if row["input"] != "" else ""
        augmented_row = {
            "meta_instruction": self.meta_instruction,
            "instruction": row["instruction"],
            "input": row["input"] + separator + formatted_tags,
            "output": row["output"] if not self.sample_tag_values else None,
            "num_tags": len(selected_tags),
            "selected_tags": selected_tags,
            "tag_values": tag_values,
            "original_input": row["input"],
            "original_output": row["output"],
            "groundtruth_tags": groundtruth_tags,
        }
        self.pbar.update()
        return augmented_row

    def sample_tags(self) -> List[str]:
        """Choose which tag to use. Note that we are not sampling the tag values like TagSampler."""
        num_tags = random.randint(1, len(self.selected_features) if self.max_num_tags < 0 else self.max_num_tags)
        selected_ids = random.sample(range(len(self.selected_features)), num_tags)
        selected_tags = [self.selected_features[i] for i in sorted(selected_ids)]
        return selected_tags

    def augment_row(self, row: dict) -> List[dict]:
        augmented_rows = []

        if self.same_tag_per_aug:
            selected_tags = self.sample_tags()

        for _ in range(self.num_aug):
            if not self.same_tag_per_aug:
                selected_tags = self.sample_tags()
            augmented_rows.append(self.augment_row_once(row, selected_tags))

        return augmented_rows

    def augment(self, df: pd.DataFrame) -> List[dict]:
        self.pbar = tqdm(total=len(df), desc="Augmenting dataset")
        augmented_data = df.apply(lambda row: self.augment_row(row.to_dict()), axis=1)
        augmented_data = [item for sublist in augmented_data for item in sublist]
        del self.pbar
        return augmented_data

    def __call__(self, *args, **kwargs):
        return self.augment(*args, **kwargs)


def load_json(path: str) -> dict:
    return json.load(open(path, "r"))


def save_json(file: dict, path: str) -> None:
    json_data = json.dumps(file, indent=4)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_data)


def pre_lftk_filter(data: dict) -> dict:
    filtered_data = [example for example in data if example["output"] != ""]
    print(f"Filtered out {len(data) - len(filtered_data)} examples pre-LFTK")
    return filtered_data


def post_lftk_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out a few outliers"""
    filtered_df = df[df["fkre"] <= 121.22]
    print(f"Filtered out {len(df) - len(filtered_df)} examples post-LFTK")
    return filtered_df


def extract(example):
    doc = nlp(example["output"])
    extracted_features = lftk.Extractor(docs=doc).extract()
    example.update(extracted_features)
    return example


def main(args):
    data = load_json(args.input_data_path)
    print(f"Original data size: {len(data)}")

    data = pre_lftk_filter(data)

    # Extract lftk features
    interm_data_path = args.interm_data_path if args.interm_data_path else args.input_data_path.replace(".json", "_with_lftk_features.json")
    if not args.use_cache or (args.use_cache and not os.path.exists(interm_data_path)):
        print("Extracting lftk features from scratch")
        num_workers = args.num_workers
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks and store the future objects
            futures = [executor.submit(extract, example) for example in data]

            # Process results as they are completed
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
                result = future.result()
                results.append(result)

        save_json(results, interm_data_path)
    else:
        print("Loading cached extracted lftk features from", interm_data_path)
        results = load_json(interm_data_path)

    df = pd.DataFrame(results)
    df = post_lftk_filter(df)

    # Collect feature ranges, means, and sigmas
    feat_ranges = {}
    feat_means = {}
    feat_sigmas = {}
    for feature in args.selected_features:
        cast = lambda x: int(x) if x.is_integer() else x
        feat_ranges[feature] = {"min": cast(df[feature].min()), "max": cast(df[feature].max())}
        feat_means[feature] = df[feature].mean()
        feat_sigmas[feature] = df[feature].std()

    # Collect feature means and sigmas after all processing steps above
    float_df = df.select_dtypes(include=["float64"])
    quantizer = Quantizer(feat_ranges, float_df.columns, args.quantization_option)
    standardizer = Standardizer(feat_means, feat_sigmas)
    tag_sampler = TagSampler(quantizer, standardizer, scale=args.scale)

    # Quantize float features
    selected_float_features = [feature for feature in args.selected_features if feature in float_df.columns]
    for feature in selected_float_features:
        df[feature] = df[feature].apply(lambda row: quantizer.do({feature: row})[feature])

    # Make dataset
    test_df = df.sample(args.test_size, random_state=args.seed)
    train_df = df.drop(test_df.index)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_augmenter = Augmenter(
        selected_features=args.selected_features,
        quantization_option=args.quantization_option,
        num_aug=args.N_train,
        max_num_tags=args.max_num_tags,
        same_tag_per_aug=args.train_same_tag_per_aug,
        sample_tag_values=args.train_sample_tag_values,
        tag_sampler=tag_sampler,
        few_shot=args.few_shot,
        augment=args.augment,
    )
    train_dataset = train_augmenter(train_df)

    test_augmenter = Augmenter(
        selected_features=args.selected_features,
        quantization_option=args.quantization_option,
        num_aug=args.N_test,
        max_num_tags=args.max_num_tags,
        same_tag_per_aug=args.test_same_tag_per_aug,
        sample_tag_values=args.test_sample_tag_values,
        tag_sampler=tag_sampler,
        few_shot=args.few_shot,
        augment=args.augment,
    )
    test_dataset = test_augmenter(test_df)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Total: {len(train_dataset) + len(test_dataset)}")

    output_data_path = args.output_data_path if args.output_data_path else "data/tagged_" + os.path.basename(args.input_data_path)

    if args.few_shot:
        few_shot_test_dataset = test_dataset

        # Find the set of demos
        from collections import Counter
        from copy import deepcopy
        def val_ok(val_indices, few_shot_test_dataset):
            # Each tag appear at least once
            # No duplicates in the demos
            cnt = Counter()
            num_no_inputs = 0
            for idx in val_indices:
                if few_shot_test_dataset[idx]["original_input"] == "":
                    num_no_inputs += 1
                new_cnt1 = deepcopy(cnt)
                for tag in few_shot_test_dataset[idx]["selected_tags"]:
                    new_cnt1[tag] += 1
                new_cnt2 = deepcopy(cnt)
                for tag in few_shot_test_dataset[idx+1]["selected_tags"]:
                    new_cnt2[tag] += 1
                cnt = new_cnt2 if len(new_cnt2) > len(new_cnt1) else new_cnt1

            if (num_no_inputs / len(val_indices) > 0.5) or any([cnt[tag] == 0 for tag in args.selected_features]):
                return False
            return True

        # I have to do it this way to make sure the resulting test set is the same with what was used in previous experiments.
        # Otherwise, I would have to rerun ChatGPT and GPT-4 evaluation with the new test set.
        # I am too broke to do so :(
        random.seed(0)
        test_indices = random.sample(range(len(few_shot_test_dataset)), args.test_size_few_shot)
        remaining_indices = [i for i in range(0, len(few_shot_test_dataset), 2) if i not in test_indices and i+1 not in test_indices]
        val_indices = random.sample(remaining_indices, args.num_demos)
        num_tries = 1
        print(f"Try finding a set of demos...")
        while not val_ok(val_indices, few_shot_test_dataset):
            num_tries += 1
            val_indices = random.sample(remaining_indices, args.num_demos)
        print("Found a set of demos after", num_tries, "tries")
        
        demo_template = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        demos = "\n\n".join([demo_template.format_map(few_shot_test_dataset[idx]) for idx in val_indices])
        for idx in test_indices:
            few_shot_test_dataset[idx]["demos"] = demos
            del few_shot_test_dataset[idx]["original_input"]
        few_shot_test_dataset = [example for idx, example in enumerate(few_shot_test_dataset) if idx in test_indices]
        
        print("### DEMOS")
        print(demos)

        metadata = {
            "num_examples": len(few_shot_test_dataset),
        }
        metadata.update(vars(args))
        metadata["feat_ranges"] = feat_ranges
        metadata["feat_means"] = feat_means
        metadata["feat_sigmas"] = feat_sigmas
        few_shot_test_dataset = {"metadata": metadata, "split": "test", "data": few_shot_test_dataset}

        postfix = f"quant_{args.quantization_option}_N_{args.N}_max_num_tags_{args.max_num_tags}"
        postfix += f"_num_demos_{args.num_demos}_test_size_{args.test_size_few_shot}"
        few_shot_test_path = output_data_path.replace(".json", f"_{postfix}_few_shot_test.json")
        save_json(few_shot_test_dataset, few_shot_test_path)
        print(f"Save few-shot test dataset to {few_shot_test_path}")
    else:
        metadata = {
            "num_examples": len(train_dataset) + len(test_dataset),
            "num_train_examples": len(train_dataset),
            "num_test_examples": len(test_dataset),
        }
        metadata.update(vars(args))
        metadata["feat_ranges"] = feat_ranges
        metadata["feat_means"] = feat_means
        metadata["feat_sigmas"] = feat_sigmas
        train_dataset = {"metadata": metadata, "split": "train", "data": train_dataset}
        test_dataset = {"metadata": metadata, "split": "test", "data": test_dataset}

        postfix = f"quant_{args.quantization_option}_N_train_{args.N_train}_N_test_{args.N_test}_max_num_tags_{args.max_num_tags}_scale_{args.scale}_new_eval"

        train_path = output_data_path.replace(".json", f"_{postfix}_train.json")
        test_path = output_data_path.replace(".json", f"_{postfix}_test.json")

        save_json(train_dataset, train_path)
        save_json(test_dataset, test_path)

        print(f"Saved data to {train_path}")
        print(f"Saved data to {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, default="data/alpaca_gpt4_data.json")
    parser.add_argument("--interm_data_path", type=str)
    parser.add_argument("--output_data_path", type=str)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--quantization_option", type=str, default="none", choices=["1to5", "1to10", "none"])
    parser.add_argument(
        "--selected_features",
        type=str,
        nargs="+",
        default=[
            "t_word",
            "n_noun",
            "n_verb",
            "n_adj",
            "t_uword",
            "n_unoun",
            "n_uverb",
            "n_uadj",
            "simp_ttr",
            "simp_noun_var",
            "simp_verb_var",
            "simp_adj_var",
            "fkre",
            "rt_average",
        ],
    )
    parser.add_argument("--test_size", default=2000)
    parser.add_argument("--max_num_tags", type=int, default=5)
    parser.add_argument("--N_train", type=int, default=1)
    parser.add_argument("--N_test", type=int, default=5)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--few_shot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_demos", type=int, default=5)
    parser.add_argument("--test_size_few_shot", type=int, default=500)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--train_same_tag_per_aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_sample_tag_values", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test_same_tag_per_aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_sample_tag_values", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
