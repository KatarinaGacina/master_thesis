import pyBigWig
import numpy as np
import random
from data.load_utils.data_loading import load_intervals_per_all_splits

if __name__ == "__main__":

    bw = pyBigWig.open("/path/atac_peaks.bw")
    intervals_positive = load_intervals_per_all_splits(
        "/path/positives_10000.bed", 
        "/path/split_0.json"
    )
    intervals_negative = load_intervals_per_all_splits(
        "/path/negatives_10000.bed", 
        "/path/split_0.json"
    )

    rng = random.Random(42)
    intervals_negative["train"] = rng.sample(
        intervals_negative["train"],
        min(len(intervals_positive["train"]), len(intervals_negative["train"]))
    )

    intervals = intervals_positive["train"]
    intervals.extend(intervals_negative["train"])

    total_zeros = 0
    total_ones = 0

    for chrom, start, end in intervals:
        label = np.nan_to_num(bw.values(chrom, start, end))

        zeros = np.sum(label == 0)
        ones = np.sum(label == 1)

        total_zeros += zeros
        total_ones += ones

    bw.close()

    print("Total zeros:", total_zeros)
    print("Total ones:", total_ones)
    print(total_zeros/total_ones)