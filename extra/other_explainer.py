import itertools
import json
import os.path
import pathlib
import sys

import matplotlib.pyplot as plt
import lime
import numpy
import numpy as np
import pandas
import scipy as sp
import shap
import torch
import transformers
import yaml
from lime.lime_text import LimeTextExplainer
import lime
import lime.lime_tabular
from sklearn.tree import DecisionTreeClassifier
from transformers import TextClassificationPipeline

from LaSSI.similarities.ClusteringTest import extract_proba_scores

def clazz_predict_probaU(transformer, tfidf_transformer, classifier, text, output_features):
    X_test_counts = transformer.transform(text)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    df = pandas.DataFrame(tfidf_transformer.transform(X_test_tfidf).todense(), columns=output_features)
    return classifier.predict_proba(df)

def tfidf_vectorizer_explainer(data, agg_scores):
    #https://medium.com/@ashwinkumar577/countvectorizer-and-tfidfvectorizer-for-beginner-ac81afef30aa
    data = [x[0]+" => "+x[1] for x in itertools.product(data, data)]
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    # https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html
    """
    @book{10.5555/3285754,
author = {Bengfort, Benjamin and Bilbro, Rebecca and Ojeda, Tony},
title = {Applied Text Analysis with Python: Enabling Language-Aware Data Products with Machine Learning},
year = {2018},
isbn = {9781491963043},
publisher = {O'Reilly Media, Inc.},
edition = {1st},
abstract = {From news and speeches to informal chatter on social media, natural language is one of the richest and most underutilized sources of data. Not only does it come in a constant stream, always changing and adapting in context; it also contains information that is not conveyed by traditional data sources. The key to unlocking natural language is through the creative application of text analytics. This practical book presents a data scientists approach to building language-aware products with applied machine learning. Youll learn robust, repeatable, and scalable techniques for text analysis with Python, including contextual and linguistic feature engineering, vectorization, classification, topic modeling, entity resolution, graph analysis, and visual steering. By the end of the book, youll be equipped with practical methods to solve any number of complex real-world problems. Preprocess and vectorize text into high-dimensional feature representations Perform document classification and topic modeling Steer the model selection process with visual diagnostics Extract key phrases, named entities, and graph structures to reason about data in text Build a dialog framework to enable chatbots and language-driven interaction Use Spark to scale processing power and neural networks to scale model complexity}
}"""
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    feat_dict = {v:k for k,v in count_vect.vocabulary_.items()}
    feat_names = [feat_dict[x] for x in range(len(feat_dict))]
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    output_features = tfidf_transformer.get_feature_names_out(feat_names)
    lool = X_train_tfidf.todense()
    df = pandas.DataFrame(lool, columns=output_features)
    from sklearn.metrics import accuracy_score
    classifier = DecisionTreeClassifier()
    classifier.fit(df, agg_scores)
    dictd = {-1:"Inconsistency",0:"Indifferent",1:"Implying"}

    X_test_counts = count_vect.transform(data)
    X_test_tfidf = pandas.DataFrame(tfidf_transformer.transform(X_test_counts).todense(), columns=output_features)
    y_pred = classifier.predict(X_test_tfidf)
    print(f"Accuracy Score: {accuracy_score(agg_scores, y_pred)}")
    explainer = lime.lime_tabular.LimeTabularExplainer(X_test_tfidf.values, feature_names=
    list(X_test_tfidf.columns),
                                                       class_names=[dictd[x] for x in classifier.classes_],
                                          mode='classification')

    def make_predictions(X_batch_text):
        X_test_counts = count_vect.transform(X_batch_text)
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)
        df = pandas.DataFrame(tfidf_transformer.transform(X_test_tfidf).todense(), columns=output_features)
        return classifier.predict(df)

    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer3 = shap.Explainer(make_predictions, masker=masker)
    X_test_text = numpy.array(data)
    shap_values3 = explainer3(X_test_text)
    with open("shap_text_plot.html", "w") as file:
        lobj = shap.text_plot(shap_values3, display=False)
        file.write(lobj)
    explainer2 = shap.KernelExplainer(classifier.predict, X_test_tfidf)
    shap_values = explainer2.shap_values(X_test_tfidf)
    shap.summary_plot(shap_values, X_test_tfidf,
                      show=False)  # .png,.pdf will also support here
    plt.savefig(f"shap_summary.svg", dpi=700)
    plt.show()

    for idx, y in enumerate(data):
        X_test_counts = count_vect.transform([y])
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)
        df = pandas.DataFrame(tfidf_transformer.transform(X_test_tfidf).todense(), columns=output_features)
        labels = (0,1,2)
        explanation = explainer.explain_instance(df.loc[0].values,
                                         classifier.predict_proba,
                                                 labels=labels )
        html = explanation.as_html()
        with open(f"lime_explanation_{idx}.html", "w") as html_file:
            html_file.write(html)
        # plt.show()

def tokenize_data(tokenizer, examples):
    return tokenizer(examples["text"], truncation=False)

def predict_string(pipe, N, x):
    assert isinstance(x, str)
    l = [0.0] * N
    for dct in pipe(x)[0]:
        l[int(dct["label"][6:])] = dct["score"] #   ---- label_encoder.inverse_transform([int(dct["label"][6:])])[0]
    return numpy.array(l)

def ffun(pipe, label_encoder, expected_labels, x):
    if isinstance(x, str):
        return numpy.array([predict_string(pipe, len(expected_labels), x)])
    elif isinstance(x, list):
        return numpy.array([predict_string(pipe, len(expected_labels), y) for y in x])ac
    else:
        return("Some ERRORRRR")
    # tv = torch.tensor([tokenizer.encode(v, padding="max_length", max_length=500, truncation=True) for v in x])
    # outputs = model(tv)[0].detach().numpy()
    # scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    # val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
    # return 0.0
    # min_score = -1
    # label = None
    l = [0.0] * len(expected_labels)
    for dct in pipe(x)[0]:
        l[int(dct["label"][6:])] = dct["score"] #   ---- label_encoder.inverse_transform([int(dct["label"][6:])])[0]
    return numpy.array([numpy.array(l)])

def distilbert_explainer(data, agg_scores):
    ### pip install 'accelerate>=0.26.0'
    data = [x[0] + " => " + x[1] for x in itertools.product(data, data)]
    dictd = {-1:"Inconsistency",0:"Indifferent",1:"Implying"}
    class_names = [dictd[x] for x in agg_scores]
    df = pandas.DataFrame({"text":data, "labels": class_names})
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['labels'].tolist())
    expected_labels = label_encoder.inverse_transform([0,1,2])
    from datasets import Dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = Dataset.from_pandas(df)
    tokenized_train = train_dataset.map(lambda x: tokenize_data(tokenizer, x), batched=True)
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

    # Define training arguments
    path = os.path.join("..", "catabolites","explain","distilbert")
    if not os.path.exists(path):
        # Load pre-trained DistilBERT model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

        # Prepare data collator for padding sequences
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=path,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch"
        )

        # Define Trainer object for training the model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_train,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train()

        # Save the trained model
        trainer.save_model(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

    explainer = LimeTextExplainer(class_names=expected_labels)
    for idx, y in enumerate(data):
        exp = explainer.explain_instance(y, lambda x: ffun(pipe, label_encoder, expected_labels, x), labels=(0,1,2))
        html = exp.as_html()
        with open(f"lime_explanation_{idx}_distilbert.html", "w") as html_file:
            html_file.write(html)
    # exit(1)
    # build a pipeline object to do predictions
    pred = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        return_all_scores=True,
    )
    explainer = shap.Explainer(pred, output_names=expected_labels)
    shap_values3 = explainer(df["text"])
    with open("shap_text_plot_distilbert.html", "w") as file:
        lobj = shap.text_plot(shap_values3, display=False)
        file.write(lobj)



if __name__ == '__main__':
    matrix_file = "/home/giacomo/Scrivania/LaSSI/catabolites/newcastle_mdpi/confusion_matrices_Logical.json"
    explanation_file = "/home/giacomo/Scrivania/LaSSI/test_sentences/orig/newcastle_mdpi.yaml"
    with open(matrix_file) as f:
        matrix = json.load(f)
    with open(explanation_file) as f:
        sentences = yaml.load(f, Loader=yaml.FullLoader)
    N = len(sentences)
    agg_scores, roc_scores = extract_proba_scores(N, 1.0, sys.float_info.epsilon, matrix)

    distilbert_explainer(sentences, agg_scores)
    tfidf_vectorizer_explainer(sentences, agg_scores)
