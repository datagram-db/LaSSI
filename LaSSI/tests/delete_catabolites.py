import os
import shutil
from pathlib import Path

def delete_files(delete_all_files=False, benchmarking=False):
    catabolites_dir = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute(), "catabolites")
    for subdir, dirs, files in os.walk(catabolites_dir):
        if subdir.split('/')[-1][0].isdigit() or not benchmarking:
            for dir in dirs:
                if dir == "viz":
                    dir_path = os.path.join(subdir, dir)
                    print(f"Deleting folder: {str(dir_path)}")
                    try:
                        shutil.rmtree(dir_path)
                    except OSError as e:
                        print(f"Error deleting {dir_path}: {e}")
            for file in files:
                if (file in ("gsmDB.txt", "datagramdb_output.json", "logical_rewriting.json",
                              "_cd.pickle", "_d.pickle", "_ec.pickle", "_eed.pickle", "_ic.pickle", "_ied.pickle") or
                    (file in ("internals.json", "internals-bin.json", "string_rep.txt", "meuDBs.json") and delete_all_files)):
                    file_path = os.path.join(subdir, file)
                    print(f"Deleting file: {str(file_path)}")
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")

if __name__ == '__main__':
    delete_files()
