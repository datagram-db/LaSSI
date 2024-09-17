import dataclasses
from typing import List

@dataclasses.dataclass()
class ScraperConfiguration:
    keywords: List[str] = dataclasses.field(default_factory=list)

    rss_feed_to_csv_name_with_day: dict = dataclasses.field(default_factory=dict)

    # RSSFeedCsv: bool = False
    # """Defines the configuration csv file containing the rss feed from the newspapers"""

    store_json: bool = False
    """Loads the articles temporarly stored into a csv file directly to the mini json database"""

    ignore_intervals: bool =False
    """Instead of scraping the articles from the biggest information gap found, it scrapes all the articles starting from the minimum date"""

    back_archive:bool = False
    """Uses the wayback engine to retrieve the RSS feed from the past"""

    min_date:str = "2019-10-01"
    """Minimum date in the yyyy-mm-dd format from which you need to scrape the articles"""

    threads:int = 10
    """The minimum amount of threads to be used in the scraping"""

    shuffle:bool =False
    """Shuffles the list used to do the crawling"""

    timeout:int=None
    """Sets the timeouts when crawling obsessively, so the scripts doesn't hang up (in seconds)"""

    reverse:bool = False
    """Gets the information from the most recent to the oldest when using the back_archive (flag needs to be activated)"""

    no_ssl:bool = True
    """Disables SSL Checking"""

    no_stop_if_parsed:bool = True
    """Stops from crawling back in time when you hit some day were some articles were already dowloaded"""