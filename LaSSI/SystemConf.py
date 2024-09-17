
def getNumberCPUs():
    import multiprocessing
    return multiprocessing.cpu_count()