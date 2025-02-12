from sqlitedict import SqliteDict
import csv

# db = rocksdb.DB("adjacency_list.db", rocksdb.Options(create_if_missing=True)) # node key and corresponding url node value

def process_conceptnet_csv(csv_file):
    db = SqliteDict("adjacency_list.db")

    with open(csv_file, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 3:
                continue

            relation_row = row[1]  # "357484	/a/[/r/ExternalURL/"
            concept_row = row[2]   # "/c/en/apple"
            wiktionary_url_row = row[3]  # "/http://fr.wiktionary.org/wiki/apple/]	10	231064	231065	0.25	{"dataset": "/d/wiktionary/fr"


            if "/r/ExternalURL" not in relation_row or not concept_row.startswith("/c/en/"):
                continue

            concept_word = concept_row.split("/")[-1]
            wiktionary_word = wiktionary_url_row.split("/")[5]

            # concept_key = f"con:{concept_word}"
            # wiktionary_key = f"wik:{wiktionary_word}"
            db[concept_word] = [wiktionary_word]

            # many conceptnet nodes could be linked to the same wiktionary node so i have to be appending to its adjacency list
            wik_list = db.get(wiktionary_word, [])
            wik_list.append(concept_word)
            db[wiktionary_word] = wik_list

    db.commit()
    db.close()