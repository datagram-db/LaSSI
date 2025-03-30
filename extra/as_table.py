
def from_matrix(file, output):
    import json
    import pandas
    with open(file) as f:
        d = json.load(f)
    pandas.DataFrame(d).to_csv(output, index=False)

if __name__ == "__main__":
    from_matrix("../catabolites/all_newcastle/confusion_matrices_Logical.json",
                "../similarity.csv")