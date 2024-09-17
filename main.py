# This is a sample Python script.
from LaSSI.LaSSI import LaSSI
from LaSSI.external_services.utilities.DatabaseConfiguration import DatabaseConfiguration
from LaSSI.external_services.web_cralwer.Scrape import Scrape
from LaSSI.external_services.web_cralwer.ScraperConfiguration import ScraperConfiguration


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # import yaml
    # nc = {
    #     "www.chroniclelive.co.uk/?service=rss": "ncl_chron_$day.csv",
    #     # "https://www.nexus.org.uk/news.xml":"nexus_$day.csv",
    #     "https://www.ncl.ac.uk/data/mobile/rss/pressoffice/pressnews/index.xml":"uni_$day.csv",
    #     "https://www.theguardian.com/uk/newcastle/rss": "guardian_$day.csv"
    # }
    # sc = ScraperConfiguration(rss_feed_to_csv_name_with_day=nc,back_archive=True,min_date="2024-09-16",threads=-1,shuffle=True,reverse=True,no_ssl=False,store_json=True)
    # Scrape(sc)

    input = ["This is a beautiful world", "The Golden Gate Bridge is fabulous!", "I went to Stratford upon Avon"]
    db_conf = DatabaseConfiguration(uname="giacomo", pw="omocaig", host="localhost", port=5432, db="conceptnet", fuzzy_dbs={"conceptnet": "/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/mini.h5_sql_input.txt",
                                                                                                                            "geonames": "/home/giacomo/projects/similarity-pipeline/submodules/stanfordnlp_dg_server/allCountries.txt_sql_input.txt"})

    pipeline = LaSSI("test1",input, db_conf)
    pipeline()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
