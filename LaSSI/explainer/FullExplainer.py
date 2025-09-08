import json
from dataclasses import dataclass

from LaSSI.structures.extended_fol.Formulae import Formula, formula_from_dict
from LaSSI.structures.extended_fol.TabularCWASemantics import TabularCWASemantics


@dataclass
class ExpansionGraphInput:
    original: int
    idx: int
    constituents: Formula
    entrypoint: int
    adj_graph: dict[str, dict[str, list[int]]]
    id_to_constituent: dict[str, Formula]

def load_expansion_graph_from_json_file(json_file):
    with open(json_file) as f:
        data = json.load(f)
    result = [None] * len(data)
    for idx, instance in enumerate(data):
        result[idx] = ExpansionGraphInput(instance["original"],
                            instance["idx"],
                            formula_from_dict(instance["constituents"]),
                            instance["entrypoint"],
                            instance["adj_graph"],
                            {k: formula_from_dict(v) for k, v in instance["id_to_constituent"].items()})
    return result
