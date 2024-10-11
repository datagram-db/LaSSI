import collections
import io

import dacite
import yaml

from LaSSI.external_services.web_cralwer.ScraperConfiguration import ScraperConfiguration

import six


def SentenceLoader(arg):
    if isinstance(arg, io.IOBase):
        arg = yaml.safe_load(arg)
    if isinstance(arg, dict):
        arg = dacite.from_dict(data_class=ScraperConfiguration, data=arg)
    if isinstance(arg, ScraperConfiguration):
        from LaSSI.external_services.web_cralwer.Scrape import Scrape
        return Scrape(arg)
    elif isinstance(arg, six.string_types):
        return [str(arg)]
    elif isinstance(arg, collections.abc.Iterable):
        return list(map(str, arg))
