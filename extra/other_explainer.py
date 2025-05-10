import itertools
import json
import sys

import matplotlib.pyplot as plt
import lime
import numpy
import pandas
import shap
import yaml
from lime.lime_text import LimeTextExplainer
import lime
import lime.lime_tabular
from sklearn.tree import DecisionTreeClassifier

from LaSSI.similarities.ClusteringTest import extract_proba_scores

def clazz_predict_probaU(transformer, tfidf_transformer, classifier, text, output_features):
    X_test_counts = transformer.transform(text)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    df = pandas.DataFrame(tfidf_transformer.transform(X_test_tfidf).todense(), columns=output_features)
    return classifier.predict_proba(df)





def mol(data, agg_scores, file):
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
        return preds

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

if __name__ == '__main__':
    matrix_file = "/home/giacomo/Scrivania/LaSSI/catabolites/newcastle_mdpi/confusion_matrices_Logical.json"
    explanation_file = "/home/giacomo/Scrivania/LaSSI/test_sentences/orig/newcastle_mdpi.yaml"
    with open(matrix_file) as f:
        matrix = json.load(f)
    with open(explanation_file) as f:
        sentences = yaml.load(f, Loader=yaml.FullLoader)
    N = len(sentences)
    agg_scores, roc_scores = extract_proba_scores(N, 1.0, sys.float_info.epsilon, matrix)
    mol(sentences, agg_scores, "lime")
    # agg_scores, roc_scores = extract_proba_scores(N, 1.0, sys.float_info.epsilon, matrix)
    # explainer = LimeTextExplainer(class_names=agg_scores)
    # exp = explainer.explain_instance(sentences, roc_scores, num_features=6, top_labels=1)
