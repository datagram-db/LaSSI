import os
import glob
import subprocess
import sys
from pathlib import Path
import re

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI
from tqdm import tqdm


def sort_by_numeric_value(file_path):
    match = re.search(r'(\d+).yaml', file_path)
    return int(match.group(1).split('.yaml')[0]) if match else 0


def get_and_run_all_sentences(folders, transformation=SentenceRepresentation.Logical, transformer='sentence-transformers/all-MiniLM-L6-v2'):
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute()
    sentences_dir = os.path.join(root_dir, "test_sentences")
    main_script_path = os.path.join(root_dir, "main.py")

    yaml_files = []
    for folder_name in folders:
        folder_path = os.path.join(sentences_dir, folder_name)
        yaml_files.extend(glob.glob(os.path.join(folder_path, "*.yaml")))

    yaml_files.sort(key=sort_by_numeric_value)
    os.chdir(os.path.dirname(os.path.abspath(main_script_path)))

    with tqdm(total=len(yaml_files), desc="Rewriting sentences") as pbar:
        for yaml_file in yaml_files:
            pbar.set_description(f"Rewriting sentences: {yaml_file.split('/')[-1]}")
            try:
                with open(os.devnull, 'w') as devnull:
                    sys.stdout = devnull
                    pipeline = LaSSI(yaml_file, "/home/campus.ncl.ac.uk/b9063849/PycharmProjects/LaSSI/connection.yaml", transformation, transformer)
                    pipeline.run()
                    pipeline.close()
                sys.stdout = sys.__stdout__
            except Exception as e:
                print(f"\nError running LaSSI for: {yaml_file}", e, file=sys.stderr)
            pbar.update(1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = ["orig"]

    all_outputs = True

    if all_outputs:
        transformations = [SentenceRepresentation.FullText, SentenceRepresentation.SimpleGraph, SentenceRepresentation.LogicalGraph, SentenceRepresentation.Logical]

        for transformation in transformations:
            if transformation == SentenceRepresentation.FullText:
                transformers = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2", "all-roberta-large-v1"]
                for transformer in transformers:
                    get_and_run_all_sentences(folders, transformation, f"sentence-transformers/{transformer}")
            else:
                get_and_run_all_sentences(folders, transformation)
    else:
        get_and_run_all_sentences(folders)
