import os
import shutil
from pathlib import Path


# Only delete GSM information as no reason to re-get meuDB
def delete_files(delete_all_files=False, benchmarking=False):
    catabolites_dir = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute(), "catabolites")
    for subdir, dirs, files in os.walk(catabolites_dir):
        if subdir.split('/')[-1][0].isdigit() or not benchmarking:
            for dir in dirs:
                if dir == "viz":
                    dir_path = os.path.join(subdir, dir)
                    print(f"Deleting folder: {dir_path}")
                    shutil.rmtree(dir_path)
            for file in files:
                if file in ("gsmDB.txt", "datagramdb_output.json") or (
                        file in ("internals.json", "internals-bin.json", "string_rep.txt", "meuDBs.json") and delete_all_files):
                    file_path = os.path.join(subdir, file)
                    print(f"Deleting file: {file_path}")
                    os.remove(file_path)


if __name__ == '__main__':
    delete_files()
