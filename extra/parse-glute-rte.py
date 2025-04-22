import itertools

import pandas
import yaml

def primal_splitting_eval_dataset():
    ## https://huggingface.co/datasets/gimmaru/glue-rte
    pd = pandas.read_parquet("hf://datasets/gimmaru/glue-rte/data/validation-00000-of-00001-76e2c209e9eab12a.parquet")
    N = 50
    pd["index"] = list(pd.index)

    idx = pd.sentence1.str.len().sort_values().index
    pd = pd.reindex(idx)
    sentences_selected = list(idx)[:N]
    idx_selected = set(pd.iloc[sentences_selected]["index"])
    with open("../test_sentences/real_data/glue-rte_index1.yaml", "w") as file:
        yaml.safe_dump(list(idx), file)
    S = set(sentences_selected)
    idx = pd.sentence2.str.len().sort_values().index
    pd = pd.reindex(idx)
    sentences_selected = list(idx)[:N]
    idx_selected = idx_selected.union(set(pd.iloc[sentences_selected]["index"]))
    with open("../test_sentences/real_data/glue-rte_index2.yaml", "w") as file:
        yaml.safe_dump(sentences_selected, file)

    pos = list()
    neg = list()
    for d in [pd[pd["index"] == x].to_dict("records")[0] for x in idx_selected]:
        if d["label"] == 0:
            pos.append({"Q": d["sentence1"], "R": d["sentence2"], "class": "entailment", "mark": False, "orig_idx": d["index"], "curr_neg_idx": len(neg)})
        else:
            neg.append({"Q": d["sentence1"], "R": d["sentence2"], "class": "not_entailment", "mark": False, "orig_idx": d["index"], "curr_pos_idx": len(pos)})
    print(f"Pos: #{len(pos)}")
    print(f"Neg: #{len(neg)}")
    with open(f"../test_sentences/real_data/glue-rte_{N}_pos_QR.yaml", "w") as file:
        yaml.safe_dump(pos, file)
    with open(f"../test_sentences/real_data/glue-rte_{N}_neg_QR.yaml", "w") as file:
        yaml.safe_dump(neg, file)

if __name__ == "__main__":
    neg_file = "/home/giacomo/projects/LaSSI/test_sentences/real_data/glue-rte_50_neg_QR.yaml"
    pos_file = "/home/giacomo/projects/LaSSI/test_sentences/real_data/glue-rte_50_pos_QR.yaml"
    with open(neg_file) as f:
        d = yaml.safe_load(f)
        neg = map(lambda x: [x["Q"], x["R"]], filter(lambda x: x["mark"], d))
        neg = itertools.chain.from_iterable(neg)
    with open(pos_file) as f:
        d = yaml.safe_load(f)
        pos = map(lambda x: [x["Q"], x["R"]], filter(lambda x: x["mark"], d))
        pos = itertools.chain.from_iterable(pos)
    data = list(itertools.chain(neg, pos))
    with open("/test_sentences/real_data/glue-rte_10_selected_QR.yaml", "w") as file:
        yaml.safe_dump(data, file)
    print(data)
