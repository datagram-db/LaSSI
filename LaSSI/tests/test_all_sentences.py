import glob
import os
import subprocess
from pathlib import Path


def get_and_run_all_sentences():
  root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute()
  sentences_dir = os.path.join(root_dir, "test_sentences")
  main_script_path = os.path.join(root_dir, "main.py")
  for folder_name in ["extension", "orig", "real_data"]:
    folder_path = os.path.join(sentences_dir, folder_name)

    yaml_files = glob.glob(os.path.join(folder_path, "*.yaml"))

    for yaml_file in yaml_files:
      print(f"Processing: {yaml_file}")
      try:
        os.chdir(root_dir)
        subprocess.run(["python", main_script_path, yaml_file], check=True)
      except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")

if __name__ == '__main__':
    get_and_run_all_sentences()
