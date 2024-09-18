# This is a sample Python script.
import signal
from dataclasses import asdict

from LaSSI.LaSSI import LaSSI
from LaSSI.external_services.utilities.DatabaseConfiguration import DatabaseConfiguration
from LaSSI.external_services.web_cralwer.Scrape import Scrape
from LaSSI.external_services.web_cralwer.ScraperConfiguration import ScraperConfiguration, serialize_configuration


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# import atexit
# def exit_handler():
#     import jpype
#     jpype.shutdownJVM()
# atexit.register(exit_handler)
# def handler(signum, frame):
#     import jpype
#     jpype.shutdownJVM()
#     print('Signal handler called with signal', signum)
# signal.signal(signal.SIGINT, handler)
# signal.signal(signal.SIGKILL, handler)
# signal.signal(signal.SIGSEGV, handler)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # import yaml
    # nc = {
    #     "www.chroniclelive.co.uk/?service=rss": "ncl_chron_$day.csv",
    #     "https://www.nexus.org.uk/news.xml":"nexus_$day.csv",
        # "https://www.ncl.ac.uk/data/mobile/rss/pressoffice/pressnews/index.xml":"uni_$day.csv",
        # "https://www.theguardian.com/uk/newcastle/rss": "guardian_$day.csv"
    # }
    # sc = ScraperConfiguration(rss_feed_to_csv_name_with_day=nc,back_archive=True,min_date="2024-09-16",threads=-1,shuffle=True,reverse=True,no_ssl=False,store_json=True)
    # serialize_configuration("conf1.yaml", sc)
    # Scrape(sc)
    # input = ["This is a beautiful world", "The Golden Gate Bridge is fabulous!", "I went to Stratford upon Avon"]
    # serialize_configuration("conf2.yaml", input)
    # db_conf = DatabaseConfiguration(uname="giacomo", pw="omocaig", host="localhost", port=5432, db="conceptnet", fuzzy_dbs={"conceptnet": "https://osf.io/download/a6yn8/",
    #                                                                                                                         "geonames": "https://osf.io/download/dprm3"})
    # serialize_configuration("connection.yaml", db_conf)
    pipeline = LaSSI("permutated.yaml", "connection.yaml")
    pipeline.run()
    pipeline.close()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
