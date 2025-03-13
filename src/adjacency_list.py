from sqlitedict import SqliteDict
from parse_conceptnet_file import Relation, CompactRelation
import csv

# db = rocksdb.DB("adjacency_list.db", rocksdb.Options(create_if_missing=True)) # node key and corresponding url node value

def process_conceptnet_csv(csv_file, db):

    with open(csv_file, "r", encoding="utf-8") as tsv:
        #reader = csv.reader(f)

        edge_label_counts = {}
        english_count = 0
        rels = []

        print("doing adjacency list initialisation")
        count = 1
        for row in csv.reader(tsv, dialect="excel-tab"):
            # if len(row) < 3:
            #     continue

            # relation_row = row[1]  # "357484	/a/[/r/ExternalURL/"
            # concept_row = row[2]   # "/c/en/apple"
            # wiktionary_url_row = row[3]  # "/http://fr.wiktionary.org/wiki/apple/]	10	231064	231065	0.25	{"dataset": "/d/wiktionary/fr"

            r = CompactRelation(row)

            #edge_label_counts[r.rel] = edge_label_counts.get(r.rel, 0) + 1
            #if r.langEnd == "en" or r.langStart == "en": english_count += 1

            # if "/r/ExternalURL" not in relation_row or not concept_row.startswith("/c/en/"):
            #     continue

            # if r.rel != "ExternalURL" or (r.langStart != "en") or (r.langEnd != "en"): continue
            # if (r.langStart != "en") or (r.langEnd != "en"): continue
            # if r.rel != "ExternalURL": continue
            if r.rel != "ExternalURL" or (r.lang != "en"): continue ## gives a count of almost 420,000
            # if r.rel != "ExternalURL" or (r.langEnd != "en"): continue ## has a count of 0

            count += 1
            rels.append(r)
            if count % 1000 == 0:
                print(count)
                # print(r.rel)
                # #print(r.langStart)
                # #print(r.langEnd)
                # print(r.lang)
                # print(r.surfaceStart)
                # print(r.surfaceEnd)

            # concept_word = concept_row.split("/")[-1]
            # wiktionary_word = wiktionary_url_row.split("/")[5]

            # concept_key = f"con:{concept_word}"
            # wiktionary_key = f"wik:{wiktionary_word}"
            db[r.surfaceStart] = [r.surfaceEnd]

            # many conceptnet nodes could be linked to the same wiktionary node so i have to be appending to its adjacency list
            wik_list = db.get(r.surfaceEnd, [])
            wik_list.append(r.surfaceStart)
            db[r.surfaceEnd] = wik_list

    print("hi lol")

    if type(db)==SqliteDict:
        db.commit()