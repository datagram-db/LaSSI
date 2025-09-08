import csv
import gzip
import json
import os.path

import io
from collections import defaultdict

from LaSSI.Parmenides.conceptnet.parse_conceptnet_file import CompactRelation


class ConceptNet:
    def __init__(self, obj, headers=None):
        if headers is None:
            headers = ["uri", "relation_id", "start_id", "end_id", "data"]
        self.obj = obj
        self.file = None
        self.headers = headers

    def __enter__(self):
        self.z = None
        if isinstance(self.obj, str):
            if self.obj.endswith(".csv"):
                self.file = open(self.obj, encoding="utf-8")
            elif self.obj.endswith(".gz"):
                self.file = gzip.open(self.obj,'rt', encoding="utf-8")
            # elif self.obj.endswith(".7z"):
            #     self.z = py7zr.SevenZipFile(self.obj, mode='r')
            #     self.file = self.z.read([self.obj[:-3]+".csv"])[self.obj[:-3]+".csv"]
            else:
                raise RuntimeError(f"Unsupported file type: {self.obj}")
        elif isinstance(self.obj, io.IOBase):
            self.file = self.obj
        else:
            raise RuntimeError(f"Unsupported file type: {type(self.obj)}")
        self.iterator = csv.reader(self.file, dialect="excel-tab")
        return self

    def __iter__(self):
        return self

    def __next__(self)->CompactRelation:
        val = next(self.iterator, None)
        if val is not None:
            return CompactRelation(val, headers=self.headers)
        raise StopIteration


    def __exit__(self, exc_type, exc_value, traceback):
         if self.file is not None:
            self.file.close()
         if self.z is not None:
             self.z.close()


def getClusters(cnls:list[ConceptNet], db_name):
    """
    process_conceptnet_csv
    """
    count = 0
    S = {"wiki", "resource", "wn31", "2012", "umbel"}
    if not os.path.exists("/data/adj_list.json"):
        db = defaultdict(set)
        for cn in cnls:
            for r in cn:
                if r.rel != "ExternalURL" or (r.lang != "en"): continue  ## gives a count of almost 420,000
                if r.surfaceEnd in S:
                    print(r.surfaceEnd)
                    continue
                count += 1
                if count % 10000 == 0:
                    print(count)
                db[r.surfaceStart].add(r.surfaceEnd)
                db[r.surfaceEnd].add(r.surfaceStart)
            print("done")
        from LaSSI.Parmenides.conceptnet.transitive_closure import floyd_warshall, build_clusters
        db = {k:sorted(list(v)) for k,v in db.items()}
        with open("/data/adj_list.json", "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=4)
    else:
        db = json.load(open("/data/adj_list.json"))

    print("floyd_warshall")
    floyd_warshall(db)
    print("build_clusters")
    from sqlitedict import SqliteDict
    clusters = SqliteDict(db_name)
    build_clusters(db, clusters)
    clusters.commit()
    clusters.close()


if __name__ == "__main__":
    with ConceptNet("/home/gyankos/ontology_integration/data/conceptnet-assertions-5.7.0.csv.gz", ["uri", "relation_id", "start_id", "end_id", "data"]) as conceptnet1:
        with ConceptNet("/home/gyankos/ontology_integration/data/edges.csv", ["id", "uri", "relation_id", "start_id", "end_id", "weight", "data"]) as conceptnet2:
            getClusters([conceptnet1, conceptnet2], "/home/gyankos/ontology_integration/data/for_transitive.sqlite")
