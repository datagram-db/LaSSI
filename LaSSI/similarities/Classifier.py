# https://aclanthology.org/2024.findings-acl.353/
class Classifier:
    def __init__(self, model=None):
        if model is None:
            model = "qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3"
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model)

    def __call__(self, premise, consequence):
        prompt = f"If {premise}, then {consequence}"
        result = self.pipe(prompt)[0]
        if (result["label"] == 1):
            score =  result["score"]/2.0+0.5
        elif (result["label"] == 0):
            score = (1-result["score"])/2.0
        else:
            raise RuntimeError("ERROR: unexpected label {}".format(result["label"]))
        print(f"Prompt: '{prompt}'. Score: {score}. Label: {result['label']}")
        return score


if __name__ == "__main__":
    pipe = Classifier()
    print(pipe("Alice skates", "Bob skates"))# class , score 0
    print(pipe("Alice Skates", "Alice does skate"))
    print(pipe("Alice skates","Alice does not skate"))
