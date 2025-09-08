#! /usr/bin/env python3
__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2023, KnoBAB"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "3.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import collections
import numpy

d = dict()

print("File reading...")
line_count = 0
to_hist = list()
traces_data = list()
nsplits = 1000


def do_bins(traces_data, data, binsize, samplefoldsize):
    rng1 = numpy.random.RandomState(1)
    rng2 = numpy.random.RandomState(2)
    yvals = list()
    bins = collections.defaultdict(list)
    bins_size = collections.defaultdict(int)
    bins_sized = collections.defaultdict(list)
    min_val = min(data)  # needed to anchor the first bin
    max_val = max(data)
    i = 0
    LL = []
    for idx, value in enumerate(data):
        bin_num = int(round(((value - min_val) / max_val) * binsize))  # integer division to find bin
        yvals.append(bin_num)
        if bin_num not in bins:
            bins[bin_num] = list()
        bins[bin_num].append(traces_data[idx])
        i = i + 1
    for k, v in bins.items():
        n = len(v)
        bins_size[n] = bins_size[n] + 1
        bins_sized[n].append(k)
    bins_size = collections.OrderedDict(bins_size)
    # print(bins_size)
    for _ in range(0, len(traces_data), samplefoldsize):
        # Sampling from the bins according to their associated frequency of items
        selected_traces = list()
        current_size = samplefoldsize
        while current_size > 0 and len(bins_size.keys())>0:
            for original_sample_bin_size in list(
                    rng1.choice(list(bins_size.keys()), p=[x / sum(bins_size.values()) for x in bins_size.values()], replace=samplefoldsize > len(bins_size),
                                size=samplefoldsize)):
                ls = bins_sized[original_sample_bin_size]
                if len(ls) ==0:
                    continue
                # bins_size[original_sample_bin_size] = bins_size[original_sample_bin_size] - 1
                bucket_id = bins_sized[original_sample_bin_size][rng2.randint(0, len(ls), 1)[0]]
                ls2 = bins[bucket_id]
                if len(ls2) == 0:
                    continue
                trace_id = bins[bucket_id].pop(rng2.randint(0, len(ls2), 1)[0])
                if len(bins[bucket_id]) == 0:
                    del bins[bucket_id]
                    bins_sized[original_sample_bin_size].remove(bucket_id)
                    if len(bins_sized[original_sample_bin_size]) == 0:
                        del bins_sized[original_sample_bin_size]
                        del bins_size[original_sample_bin_size]
                selected_traces.append(trace_id)
                current_size -= 1
                if current_size == 0:
                    break
        # print(len(selected_traces))
        LL.append(selected_traces)
    return LL

with open("/home/giacomo/Scaricati/result.txt", "r") as file1:
    count = 0
    dict_elems = {}
    for line in file1:
        trace = line.strip().split(" ")
        traces_data.append(trace[0])
        to_hist.append(int(trace[1]))
        line_count = line_count + 1
        dict_elems[trace[0]] = count
        count += 1
    min_val = min(to_hist)
    max_val = max(to_hist)
    x = range(line_count)
    y = do_bins(traces_data, to_hist, 10, 117)
    S = set()
    for z in y:
        S = S.union(z)
        print(S)
        # import matplotlib.pyplot as plt
        # plt.bar(list(S), list(map(lambda k: to_hist[dict_elems[k]], S)), color='g')
        # plt.show()


