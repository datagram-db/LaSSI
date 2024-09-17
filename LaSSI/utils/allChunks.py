import numpy


def chunk(x, n):
    return zip(*(x[i:] for i in range(n)))


def allChunks(l):
    l = list(l)
    s = list()
    for i in range(len(l)+1):
        if i > 1:
            for x in chunk(l, i):
                s.append(x)
    return s
