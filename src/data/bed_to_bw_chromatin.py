import pyBigWig

from load_utils.data_loading import load_genome_sizes
from load_utils.data_loading import load_intervals_as_dict

def create_bigwig_from_bed_chromatin(chromatin_bed, chrom_sizes_filename, output_path):

    chrom_sizes_dict = load_genome_sizes(chrom_sizes_filename)
    intervals = load_intervals_as_dict(chromatin_bed, sort=True)

    bw = pyBigWig.open(output_path, "w")

    header = [(chrom, length) for chrom, length in chrom_sizes_dict.items()]
    bw.addHeader(header)

    for chrom in chrom_sizes_dict.keys():

        chrom_intervals = intervals.get(chrom, [])
        chrom_len = chrom_sizes_dict.get(chrom, 0)

        starts = [max(0, start) for start, end in chrom_intervals]
        ends = [min(chrom_len, end) for start, end in chrom_intervals]
        values = [1.0] * len(starts)

        bw.addEntries(chroms=[chrom]*len(starts), starts=starts, ends=ends, values=values)
        
        print(f"Added data for {chrom}")

    bw.close()
    print("BigWig file successfully created.")


if __name__ == "__main__":
    genome_sizes_filename = "/path/assembly.fasta.gz.fai"
    chromatin_bed = "/path/atac_peaks_overlapped.bed"
    output_bw = "/path/atac_peaks.bw"

    create_bigwig_from_bed_chromatin(chromatin_bed, genome_sizes_filename, output_bw)