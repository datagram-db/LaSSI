import os
import glob
import subprocess
from pathlib import Path
from tqdm import tqdm

def get_and_run_all_sentences():
  root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute()
  sentences_dir = os.path.join(root_dir, "test_sentences")
  main_script_path = os.path.join(root_dir, "main.py")

  yaml_files = []
  for folder_name in ["extension", "orig", "real_data"]:
    folder_path = os.path.join(sentences_dir, folder_name)
    yaml_files.extend(glob.glob(os.path.join(folder_path, "*.yaml")))
  yaml_files.sort()

  with tqdm(total=len(yaml_files), desc="Rewriting sentences") as pbar:
    for yaml_file in yaml_files:
      pbar.set_description(f"Rewriting sentences: {yaml_file.split('/')[-1]}")
      try:
        os.chdir(root_dir)
        with open(os.devnull, 'w') as devnull:
          subprocess.run(["python", main_script_path, yaml_file], check=True, stdout=devnull, stderr=devnull)
      except subprocess.CalledProcessError as e:
        print(f"\nError running LaSSI for: {yaml_file}")
      pbar.update(1)

if __name__ == '__main__':
    get_and_run_all_sentences()