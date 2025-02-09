import sys

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI

if __name__ == '__main__':
    dataset_name = "test_sentences/orig/alice_bob.yaml"
    fuzzyDBs = "connection_giacomo.yaml"

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        fuzzyDBs = sys.argv[2]

    pipeline = LaSSI(dataset_name, fuzzyDBs) #, SentenceRepresentation.FullText)
    pipeline.run()
    pipeline.close()
