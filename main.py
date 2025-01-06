import sys
from LaSSI.LaSSI import LaSSI

if __name__ == '__main__':
    dataset_name = "test_sentences/real_data/part9.yaml"
    fuzzyDBs = "connection.yaml"

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        fuzzyDBs = sys.argv[2]

    pipeline = LaSSI(dataset_name, fuzzyDBs)
    pipeline.run()
    pipeline.close()
