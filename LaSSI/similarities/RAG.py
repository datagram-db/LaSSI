import os
import sys

from ragatouille import RAGPretrainedModel

#

def rag(name_model, experiment_name, sentences):
    sentences = sentences
    assert name_model.startswith("RAG#")
    actual_name = name_model[4:]
    rag = RAGPretrainedModel.from_pretrained(actual_name)
    N = len(sentences)
    path = f".ragatouille/colbert/indexes/{experiment_name}/"
    # self.path = os.path.join(main_catabolites_folder, experiment_name, "ragatouille", self.actual_name.replace("/", "_").replace("\\","_"))
    if not os.path.exists(path):
        rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        rag.index(collection=sentences,
            document_ids=[str(x) for x in range(N)],
            index_name=experiment_name,
            max_document_length=180,
            split_documents=True)
    rag = RAGPretrainedModel.from_index(path)
    MM = -1
    mm = sys.float_info.max
    L = []
    for sentence in sentences:
        docs = rag.search(sentence, k=N)
        result = [(int(x['document_id']), x['score']) for x in docs]
        result.sort(key=lambda x: x[0])
        result = [x[1] for x in result]
        MM = max(MM, max(result))
        mm = min(mm, min(result))
        L.append(result)
    return [[(x-mm)/(MM-mm) for x in l] for l in L]