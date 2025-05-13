import sys

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI

if __name__ == '__main__':
    dataset_name = "test_sentences/orig/newcastle_mdpi3.yaml"
    fuzzyDBs = "connection_giacomo.yaml"
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        fuzzyDBs = sys.argv[2]

    # Changes:
    # 1. SentenceRepresentation now have DisabledAdHoc variants (except from the Embedders)
    # 2. New embedding system, RAG#colbert-ir/colbertv2.0. To compare other systems for question answering, which is the thing we are targeting, I provided references to ColBERTv2, which is not only using embedding based, but also with "RAG#colbert-ir/colbertv2.0"
    # 3. For exploiting the implication classifier, I used a very recent paper also avialable through HuggingFace: Log#"Log#qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3"
    pipeline = LaSSI(dataset_name, fuzzyDBs, SentenceRepresentation.Logical)
    pipeline.run()
    pipeline.close()
