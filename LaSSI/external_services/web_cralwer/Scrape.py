from LaSSI.external_services.web_cralwer.ScraperConfiguration import ScraperConfiguration


def Scrape(args:ScraperConfiguration):
    from LaSSI.external_services.web_cralwer.scrape_script import scrape_by_keyword
    L = scrape_by_keyword(args)
    ## TODO: also parse the dumped HTML files in the "extra" folder
    return L