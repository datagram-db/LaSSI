import edge_mapping
import adjacency_list
import transitive_closure
import parse_conceptnet_file
import wiktionary_json_extract
import json
from config import config
import csv


def generate(conceptnet_path, wiktionary_path, test_limit: -1):

    print("hi")
    adjacency_list.process_conceptnet_csv(conceptnet_path)

    transitive_closure.build_closures()
    transitive_closure.build_clusters()

    with open(config["result_file"], "w") as tsv:
        wr = csv.writer(tsv, delimiter="\t")
        wr.writerow(["source", "relation", "target"])

        print("making result file")
        count = 0
        for triplet in parse_conceptnet_file.get_triplets(conceptnet_path, lang="en"):
            (source, edge_label, target) = triplet
            #wr.writerow((transitive_closure.get_node(source), edge_mapping.get_edge(relation), transitive_closure.get_node(target)))
            wr.writerow((transitive_closure.get_node(source), edge_mapping.get_edge(edge_label),
                         transitive_closure.get_node(target)))
            count += 1
            if count % 1000 == 0: print(count)
            if count == test_limit: break

        wik_json = json.load(wiktionary_path)
        for triplet in wiktionary_json_extract.extract_information(wik_json, language_code="en"):
            (source, edge_label, target) = triplet
            source = transitive_closure.get_node(source.split("#")[0])
            target = transitive_closure.get_node(target.split("#")[0])

            wr.writerow((source, edge_mapping.get_edge(edge_label),target))






