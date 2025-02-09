import json
import os.path
import pickle
from collections import defaultdict
from pathlib import Path

import yaml

def represents_int(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True


class CollectEvidenceFromDataset():
    def __init__(self):
        self.nmods = set()
        self.acl_relcl = set()
        self.snd = defaultdict(set)
        self.mark = defaultdict(set)
        self.Props = collect_proposition_names()
        self.propInSentence = defaultdict(set)

    def from_dataset(self, ya):
        f = os.path.join("../catabolites", Path(ya).stem, "internals.json")
        # f = "catabolites/all_concept/internals.json"
        # f = "catabolites/newcastle/internals.json"
        with open(f, "rb") as f:
            data = pickle.load(f)
        with open(ya, "r") as f:
            ls = yaml.full_load(f)
        assert ls is not None
        assert data is not None
        assert len(ls) == len(data)

        for idx, (sentence, x) in enumerate(zip(ls, data)):
            for mark, val in x.sentences.extract_properties(lambda k,v: represents_int(k) or k == "mark" or v in self.Props):
                if mark in {"begin", "end", "lemma"}:
                    continue
                try:
                    float(mark)
                    self.mark["_float"].add(val)
                except:
                    self.mark[mark].add(val)
                self.propInSentence[val].add(sentence)
            self.snd[sentence].update({v for k,v in x.sentences.extract_properties(lambda k, v: k == "nmod")})
            self.nmods.update(self.snd[sentence])
            self.acl_relcl.update({v for k,v in x.sentences.extract_properties(lambda k, v: k == "acl_relcl")})

    def finalize(self):
        self.nmods = {(x.kernel.source.to_string(), x.kernel.target.to_string()) for x in self.nmods if hasattr(x, "kernel")}
        self.snd = {k: [(x.kernel.source.to_string(), x.kernel.target.to_string()) if hasattr(x, "kernel") else x for x in v]
               for k, v in self.snd.items() if len(v) > 0}
        self.acl_relcl = {(x.kernel.source.to_string(), x.kernel.target.to_string()) for x in self.acl_relcl if
                     hasattr(x, "kernel")}

def collect_proposition_names():
    data = None
    with open("../raw_data/prepositions.json", "r") as f:
        data = json.load(f)
    data.pop("__sources")
    Props = set()
    Props.update(data.keys())
    return Props

if __name__ == "__main__":
    data = None
    ya = "/home/giacomo/projects/LaSSI/test_sentences/all_concept.yaml"
    cefd = CollectEvidenceFromDataset()
    cefd.from_dataset("test_sentences/all_concept.yaml")
    cefd.from_dataset("test_sentences/orig/all_newcastle.yaml")
    cefd.from_dataset("test_sentences/orig/alice_bob.yaml")
    cefd.from_dataset("test_sentences/to_be_implemented/resolving_variables.yaml")
    cefd.finalize()
    print(cefd.nmods)
    # print(json.dumps({k: list(v) for k,v in dict(cefd.propInSentence).items()}))
    # print(json.dumps({k: list(v) for k,v in dict(cefd.mark).items()}))
