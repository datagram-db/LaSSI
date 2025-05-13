from LaSSI.LaSSI import LaSSI
from LaSSI.Configuration import SentenceRepresentation
from LaSSI.explainer.LaSSIExplainer import LaSSIExplainer
from LaSSI.explainer.ReportBuilder import ReportBuilder

if __name__ == '__main__':
    ### Setting up the services as per the main LaSSI pipeline
    dataset_name = "/home/giacomo/Scrivania/LaSSI/test_sentences/orig/newcastle_mdpi2.yaml"
    fuzzyDBs="/home/giacomo/Scrivania/LaSSI/connection_giacomo.yaml"
    LaSSIExplainer.start_up_services(fuzzyDBs)
    ## Running the full pipeline to retrieve the information using the ids.
    pipeline = LaSSI(dataset_name, fuzzyDBs, SentenceRepresentation.Logical, run_ex_post=False, useId=True)
    pipeline.run()
    ## Expanding using the IDs: this will blow up the number of the rules, but it will generate elements
    ## with provenance
    exp = LaSSIExplainer(dataset_name)
    exp.dump_explanation(11, 11)
    exp.dump_explanation(11, 2)
    exp.dump_explanation(2, 11)