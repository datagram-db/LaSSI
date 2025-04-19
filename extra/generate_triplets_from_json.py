import json
import os
import pickle
import sys

def make_output_as_triplets(sentences_or_file):
    if isinstance(sentences_or_file, list):
        if all(map(lambda x: isinstance(x, str), sentences_or_file)):
            data = [json.loads(x) for x in sentences_or_file]
        else:
            data = sentences_or_file
    elif isinstance(sentences_or_file, str) and os.path.isfile(sentences_or_file):
        with open(sentences_or_file, "r") as gsmdb:
            data = json.load(gsmdb)
    else:
        data = json.loads(str(sentences_or_file))
    LL = []
    for graph in data:
        D = dict()
        for entry in graph:
            D[entry["id"]] = entry["xi"][0] if len(entry["xi"]) >= 1 else ""
        L = []
        for entry in graph:
            for edge in entry["phi"]:
                src = edge["score"]["parent"]
                label = edge["score"]["label"].strip()
                dst = edge["score"]["child"]
                L.append((D[src], label, D[dst]))
        LL.append(set(L))
    return LL


if __name__ == "__main__":
    assert len(sys.argv) > 1
    file = sys.argv[1]
    data = None
    with open(file, "r") as gsmdb:
        data = json.load(gsmdb)
    LL = []
    for graph in data:
        D = dict()
        for entry in graph:
            D[entry["id"]] = entry["xi"][0] if len(entry["xi"]) >= 1 else ""
        L = []
        for entry in graph:
            for edge in entry["phi"]:
                src = edge["score"]["parent"]
                label = edge["score"]["label"].strip()
                dst = edge["score"]["child"]
                L.append((D[src], label, D[dst]))
        print(L)
        LL.append(set(L))
    if len(sys.argv)>2:
        target = sys.argv[2]
        with open(target, "wb") as gsmdb:
            pickle.dump(LL, gsmdb, pickle.HIGHEST_PROTOCOL)