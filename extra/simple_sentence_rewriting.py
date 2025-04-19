import json
import os
import pickle
import sys

import yaml
from StanfordNLPExtractor.OldWrapper import OldWrapper

from extra.generate_triplets_from_json import make_output_as_triplets

if __name__ == "__main__":
    file = "/home/giacomo/projects/LaSSI/test_sentences/orig/alice_bob.yaml"
    visualizer = "/home/giacomo/projects/LaSSI/catabolites/tmp"
    graph = file[:-5] + ".graph"
    with open(graph, "rb") as f:
        expected = pickle.load(f)
    # sentences = "Newcastle city center does not have traffic but Newcastle has traffic"
    with open(file, "r") as sentences_f:
        sentences = yaml.safe_load(sentences_f)
    data = "/home/giacomo/projects/LaSSI/catabolites/tmp_output.txt"
    ow = OldWrapper.getInstance()
    gsm_db = ow.generateGSMDatabase(sentences)
    with open(data, "w") as gsmdb:
        gsmdb.write(gsm_db)

    from PyDatagramDB import DatagramDB
    d = DatagramDB(data,
                   "/home/giacomo/projects/LaSSI/LaSSI/resources/gsm_query.txt",
                   visualizer ,
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
    count = 1
    for (x,y) in zip(expected, LL):
        for x_item in x:
            if (not x_item[1].startswith("does ")):
                if not x_item in y:
                    raise RuntimeError(f"Error: {x_item} not contained in {y}")
            else:
                if not (x_item[0], x_item[1][5:], x_item[2]) in y:
                    raise RuntimeError(f"Error: {x_item} not contained in {y}")
        print(f"Sentence #{count}: OK!")
        count += 1