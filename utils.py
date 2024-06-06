import os
import io
import json
from typing import List, Union, Any, Dict
import lftk

import copy
from copy import deepcopy

Tag = Dict[str, Union[int, float]]
DataPoint = Dict[str, Any]


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_lftk_mappings():
    lftk_key2id = {}
    lftk_id2key = {}
    for idx, feature in enumerate(lftk.search_features()):
        lftk_key2id[feature["key"]] = idx
        lftk_id2key[idx] = feature["key"]
    return lftk_key2id, lftk_id2key


def get_lftk_feature_types():
    data_types = jload("lftk_data_types.json")
    data_types = {k: eval(v) for k, v in data_types.items()}
    return data_types


def get_lftk_float_features():
    feature_types = get_lftk_feature_types()
    return set([k for k, v in feature_types.items() if v == float])


def get_lftk_int_features():
    feature_types = get_lftk_feature_types()
    return set([k for k, v in feature_types.items() if v == int])


def make_tag_description(selected_features, quantization_option="1to5"):
    # TODO: make this more general
    tag_description = (
        "[t_word] for the total number of words; "
        "[n_noun] for the total number of nouns; "
        "[n_verb] for the total number of verbs; "
        "[n_adj] for the total number of adjectives; "
        "[t_uword] for the total number of unique words; "
        "[n_unoun] for the total number of unique nouns; "
        "[n_uverb] for the total number of unique verbs; "
        "[n_uadj] for the total number of unique adjectives; "
        "[simp_ttr] for the simple type-token ratio; "
        "[simp_noun_var] for simple noun variation; "
        "[simp_verb_var] for simple verb variation; "
        "[simp_adj_var] for simple adjective variation; "
        "[fkre] for the Flesch-Kincaid Reading Ease; "
        "[rt_average] for the average reading time."
    )
    return tag_description


def make_tag_description_old(selected_features, quantization_option="1to5"):
    lftk_feature_map = lftk.utils.get_feature_map(lftk.lftk.FEATURE_MAP_PATH)
    float_features = get_lftk_float_features()
    tag_description = ""
    for feat in selected_features:
        tag_description += f"- {feat}: {' '.join(lftk_feature_map[feat]['name'].split('_'))}"
        if feat in float_features:
            if quantization_option == "1to5":
                tag_description += " (from 1 to 5)"
            elif quantization_option == "1to10":
                tag_description += " (from 1 to 10)"
        tag_description += "\n"
    return tag_description.strip()


def relabel(example: DataPoint, new_tags: Tag, new_output: str = None) -> DataPoint:
    """Relabel the tags of the example."""
    relabeled_example = copy.deepcopy(example)
    old_formmated_tags = " ".join([f"[{tag}: {tag_value}]" for tag, tag_value in zip(example["selected_tags"], example["tag_values"])])
    new_formmated_tags = " ".join([f"[{tag}: {tag_value}]" for tag, tag_value in new_tags.items()])

    relabeled_example["input"] = example["input"].replace(old_formmated_tags, new_formmated_tags)
    relabeled_example["num_tags"] = len(new_tags)
    relabeled_example["selected_tags"] = list(new_tags.keys())
    relabeled_example["tag_values"] = list(new_tags.values())

    relabeled_example["output"] = new_output

    return relabeled_example


def example2tag(example: DataPoint) -> Tag:
    return dict(zip(example["selected_tags"], example["tag_values"]))


def examples2tags(examples: List[DataPoint]) -> List[Tag]:
    return [example2tag(example) for example in examples]


def has_derived_features(tags: dict) -> bool:
    if (
        "simp_ttr" in tags
        or "simp_noun_var" in tags
        or "simp_verb_var" in tags
        or "simp_adj_var" in tags
        or "rt_average" in tags
    ):
        return True
    return False

def is_valid(tags: dict) -> dict:
    # Given a tag vector, return if it is valid

    supported_features = {
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
    }
    if not set(tags.keys()).issubset(supported_features):
        raise ValueError("tags should only contain supported features")

    if has_derived_features(tags):
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

    if tags["fkre"] > 121.22:
        return False

    return True


def safe_div(a, b):
    return a / b if b != 0 else 0


def compute_derived_features(tags: dict, ndigits=2) -> dict:
    # Compute derived features from the given tags
    derived_features = {}

    derived_features["simp_ttr"] = round(safe_div(tags["t_uword"], tags["t_word"]), ndigits)
    derived_features["simp_noun_var"] = round(safe_div(tags["n_unoun"], tags["n_noun"]), ndigits)
    derived_features["simp_verb_var"] = round(safe_div(tags["n_uverb"], tags["n_verb"]), ndigits)
    derived_features["simp_adj_var"] = round(safe_div(tags["n_uadj"], tags["n_adj"]), ndigits)
    derived_features["rt_average"] = round(safe_div(tags["t_word"], 240), ndigits)

    res = deepcopy(tags)
    res.update(derived_features)
    return res


def is_foundational_feature(feature: str) -> bool:
    return feature in {
        "t_word",
        "n_noun",
        "n_verb",
        "n_adj",
        "t_uword",
        "n_unoun",
        "n_uverb",
        "n_uadj",
        "fkre",
    }