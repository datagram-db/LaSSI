import sys

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI

if __name__ == '__main__':
    dataset_name = "test_sentences/benchmarking/200.yaml"
    fuzzyDBs = "connection.yaml"
    transformation = SentenceRepresentation.Logical
    transformer = 'sentence-transformers/all-MiniLM-L6-v2'

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        fuzzyDBs = sys.argv[2]

    pipeline = LaSSI(dataset_name, fuzzyDBs, transformation, transformer)
    pipeline.run()
    pipeline.close()
