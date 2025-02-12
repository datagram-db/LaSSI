import edge_mapping
import adjacency_list
import transitive_closure
import parse_conceptnet_file
from config import config
import csv


def generate(conceptnet_path, wiktionary_path, test_limit: -1):

    adjacency_list.process_conceptnet_csv(conceptnet_path)

    transitive_closure.build_closures()
    transitive_closure.build_clusters()

    with open(config["result_file"], "w") as tsv:
        wr = csv.writer(tsv, delimiter="\t")
        wr.writerow(["source", "relation", "target"])

        count = 0
        for triplet in parse_conceptnet_file.get_triplets(conceptnet_path, lang="en"):
            (source, relation, target) = triplet
            wr.writerow((transitive_closure.get_node(source), edge_mapping.get_edge(relation), transitive_closure.get_node(target)))
            count += 1
            if count == test_limit: break
        ### another loop but for wiktionary:



