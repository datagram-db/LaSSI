import os
import shutil
from pathlib import Path
from LaSSI.tests.test_all_sentences import get_and_run_all_sentences


# Only delete GSM information as no reason to re-get meuDB
def delete_files():
  catabolites_dir = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute(), "catabolites")
  for subdir, dirs, files in os.walk(catabolites_dir):
    for dir in dirs:
      if dir == "viz":
        dir_path = os.path.join(subdir, dir)
        print(f"Deleting folder: {dir_path}")
        shutil.rmtree(dir_path)
    for file in files:
      if file in ("gsmDB.txt", "datagramdb_output.json", "string_rep.txt"):
        file_path = os.path.join(subdir, file)
        print(f"Deleting file: {file_path}")
        os.remove(file_path)

if __name__ == '__main__':
    delete_files()