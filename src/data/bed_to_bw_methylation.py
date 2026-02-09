import pyBigWig

from load_utils.data_loading import load_genome_sizes, load_methyl_data_as_dict

def create_bigwig_from_bed_methylation(methylation_bed, chrom_sizes_file, output_path):
    chrom_sizes = load_genome_sizes(chrom_sizes_file)
    m_dict, h_dict = load_methyl_data_as_dict(methylation_bed)

    bw = pyBigWig.open(output_path, "w")

    header = [(chrom, length) for chrom, length in chrom_sizes.items()]
    bw.addHeader(header)
    
    for chrom in chrom_sizes.keys():
        print(f"Started with chrom: {chrom}")

        all_intervals = []

        for strand in m_dict.get(chrom, {}):
            for start, end in m_dict[chrom][strand]:
                all_intervals.append((int(start), int(end), 1.0 if strand == "+" else -1.0))

        for strand in h_dict.get(chrom, {}):
            for start, end in h_dict[chrom][strand]:
                all_intervals.append((int(start), int(end), 2.0 if strand == "+" else -2.0))

        if not all_intervals:
            continue

        all_intervals = sorted(all_intervals, key=lambda x: (x[0], x[1]))

        for start, end, val in all_intervals:
            if (start < 0):
                print("start")
            if (end > chrom_sizes[chrom]):
                print("end")
            if (end - start <= 0):
                print("razlika")

        for i in range(1, len(all_intervals)):
            prev_start, prev_end, prev_val = all_intervals[i-1]
            curr_start, curr_end, curr_val = all_intervals[i]

            if curr_start < prev_end:
                print(f"Overlap detected: ({prev_start}, {prev_end}, {prev_val}) and ({curr_start}, {curr_end}, {cur_val})")

        starts, ends, vals = zip(*all_intervals)
        bw.addEntries([chrom]*len(starts), list(starts), ends=list(ends), values=list(vals))

    bw.close()

if __name__ == "__main__":
    genome_sizes_filename = "/path/assembly.chrom.sizes"
    methylation_bed = "/path/pileup_C.bed"
    output_bw = "/path/methylation.bw"

    create_bigwig_from_bed_methylation(methylation_bed, genome_sizes_filename, output_bw)