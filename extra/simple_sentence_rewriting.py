import json
import os
import pickle
import sys
from pathlib import Path

import yaml
from StanfordNLPExtractor.OldWrapper import OldWrapper

from extra.generate_triplets_from_json import make_output_as_triplets

def as_set_of_triplets(file, limit = None):
    p = Path(file)
    parent_folder = p.parent.absolute()
    file_name = p.stem
    visualizer = f"/home/giacomo/projects/LaSSI/catabolites/{file_name}_tmp"
    data = f"/home/giacomo/projects/LaSSI/catabolites/{file_name}_gsm.txt"
    # sentences = "Newcastle city center does not have traffic but Newcastle has traffic"
    with open(file, "r") as sentences_f:
        sentences = yaml.safe_load(sentences_f)
    if limit is not None:
        sentences = sentences[:limit]
        data = f"/home/giacomo/projects/LaSSI/catabolites/{file_name}{limit}_gsm.txt"
        visualizer = f"/home/giacomo/projects/LaSSI/catabolites/{file_name}{limit}_tmp"
        split = os.path.join(parent_folder, f"{file_name}{limit}.yaml")
        with open(split, "w") as sentences_f:
            yaml.safe_dump(sentences, sentences_f)
    if not os.path.exists(visualizer):
        Path(visualizer).mkdir(parents=True, exist_ok=True)
    ow = OldWrapper.getInstance()
    gsm_db = ow.generateGSMDatabase(sentences)
    with open(data, "w") as gsmdb:
        gsmdb.write(gsm_db)

    from PyDatagramDB import DatagramDB
    d = DatagramDB(data,
                   "/home/giacomo/projects/LaSSI/LaSSI/resources/gsm_query.txt",
                   visualizer,
                   isSerializationFull=True,
                   opt_data_schema="pos\nSizeTAtt\nbegin\nSizeTAtt\nend\nSizeTAtt")
    d.run()
    L = []
    n = len(sentences)
    for result_graph_file in map(lambda x: os.path.join(visualizer, str(x), "result.json"), range(n)):
        with open(result_graph_file, "r") as f:
            raw_json_graph = json.load(f)
            L.append(raw_json_graph)

    LL = make_output_as_triplets(L)

def test(file):
    LL = as_set_of_triplets(file)
    graph = file[:-5] + ".graph"
    with open(graph, "rb") as f:
        expected = pickle.load(f)
    count = 1
    for (x, y) in zip(expected, LL):
        for x_item in x:
            if (not x_item[1].startswith("does ")):
                if not x_item in y:
                    raise RuntimeError(f"Error: {x_item} not contained in {y}")
            else:
                if not (x_item[0], x_item[1][5:], x_item[2]) in y:
                    raise RuntimeError(f"Error: {x_item} not contained in {y}")
        print(f"Sentence #{count}: OK!")
        count += 1

if __name__ == "__main__":
    file = "/home/giacomo/projects/LaSSI/test_sentences/real_data/glue-rte_answers.yaml"
    LL = as_set_of_triplets(file, 50)
    with open("triplets.json", "w") as f:
        f.write(json.dumps(LL, indent=4))