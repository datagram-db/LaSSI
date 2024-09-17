__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox, Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"
import stanza

# from crawltogsm.write_to_log import write_to_log
# from crawltogsm.write_to_log import write_to_log


class StanzaService:
    _instance = None

    nlp_token = None
    nlp = None
    stNLP = None

    def __init__(self):
        if self.nlp is None:
            # write_to_log(None, "Initialising Stanza Pipeline...")
            self.nlp = stanza.Pipeline(lang='en')

        if self.nlp_token is None:
            # write_to_log(None, "Initialising Stanza Tokenization...")
            self.nlp_token = stanza.Pipeline(lang='en', processors='tokenize')

        if self.stNLP is None:
            self.stNLP = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='en')

    def __new__(cls):
        if cls._instance is None:
            # write_to_log(None, "Initialising Stanza...")
            stanza.download('en',  processors='tokenize,mwt,pos,lemma', verbose=False)
            cls._instance = super(StanzaService, cls).__new__(cls)

        return cls._instance
