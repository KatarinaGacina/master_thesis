import pandas as pd
from data.load_utils.data_loading import load_genome_sizes

if __name__ == "__main__":

    input_csv = "/path/per_gene_counts_with_coords.csv"
    output_bed = "/path/gene_positive_100.bed"

    df_org = pd.read_csv(input_csv)
    df = df_org[df_org["total_counts"] > 100].copy()

    fai_file = "/path/assembly.gz.fai"
    chrom_sizes = load_genome_sizes(fai_file)

    start_shift = 20000
    rest_length = 79968
    
    with open(output_bed, 'w') as output_bed:
        for _, row in df.iterrows():
            chrom = row['chrom']
            start = row['start']
            end = row['end']
            strand = row['strand']

            if strand == "+":
                start_d = max(0, row['start'] - start_shift)
                end_d = min(row['start'] + rest_length, chrom_sizes[chrom])
            else:
                start_d = max(0, row['end'] - rest_length)
                end_d = min(row['end'] + start_shift, chrom_sizes[chrom])

            diff = (start_shift + rest_length) - (end_d - start_d)
            if diff > 0:
                if start_d == 0:
                    end_d += diff
                else:
                    start_d -= diff

            output_bed.write(f"{chrom}\t{start_d}\t{end_d}\t{strand}\t{start}\t{end}\n")
    
    print(f"Positive intervals saved to {output_bed} with 6 columns.")