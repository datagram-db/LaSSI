import json
from collections import defaultdict


class VisitSentence:
    def __init__(self):
        self.d = defaultdict(set)

    def visit_sentence(self, sentence):
        if sentence is None:
            return
        if sentence.get("type", None) == "SENTENCE":
            if "id" in sentence:
                id = sentence["id"]
                if "min" in sentence:
                    min = int(sentence["min"])
                    if "max" in sentence and min != -1:
                        max = int(sentence["max"])
                        if max != -1:
                            self.d[id].add((min,max))
            if "properties" in sentence:
                    for val in sentence["properties"].values():
                        for x in val:
                            self.visit_sentence(x)
        if "entities" in sentence:
            for entity in sentence["entities"]:
                self.visit_sentence(entity)
        if sentence.get("kernel", None) is not None:
            self.visit_sentence(sentence["kernel"]["source"])
            self.visit_sentence(sentence["kernel"]["target"])
            self.visit_sentence(sentence["kernel"]["edgeLabel"])
        else:
            id = sentence["id"]
            if "min" in sentence:
                min = int(sentence["min"])
                if "max" in sentence and min != -1:
                    max = int(sentence["max"])
                    if max != -1:
                        self.d[id].add((min, max))
            if "properties" in sentence:
                if "begin" in sentence:
                    min = int(sentence["begin"])
                    if "end" in sentence and min != -1:
                        max = int(sentence["end"])
                        if max != -1:
                            self.d[id].add((min, max))
                if "extra" in sentence:
                    self.visit_sentence(sentence["extra"])






if __name__ == "__main__":
    file = "/home/giacomo/Scrivania/LaSSI/catabolites/newcastle_mdpi2/internals.json"
