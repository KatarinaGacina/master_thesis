import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("/path/per_gene_counts_with_coords.csv")

    filtered = df[df["total_counts"] > 1000].copy()
    
    print(len(df))
    print(len(filtered))
    print()

    count_positive = (df["strand"] == "+").sum()
    count_negative = (df["strand"] == "-").sum()

    print("Number positive", count_positive)
    print("Number negative", count_negative)
    print()

    count_positive_filtered = (filtered["strand"] == "+").sum()
    count_negative_filtered = (filtered["strand"] == "-").sum()

    print("Number positive filtered", count_positive_filtered)
    print("Number negative filtered", count_negative_filtered)
    print()

    df["start"] = pd.to_numeric(df["start"], errors='coerce')  # 'coerce' will turn invalid values to NaN
    df["end"] = pd.to_numeric(df["end"], errors='coerce')

    df["length"] = df["end"] - df["start"]
    filtered["length"] = filtered["end"] - filtered["start"]

    max_length = df["length"].max()
    max_index = df["length"].idxmax()
    print(f"Max length: {max_length}, at index: {max_index}")
    #print(print(df.loc[max_index]))

    max_length_filtered = filtered["length"].max()
    max_index_filtered = filtered["length"].idxmax()

    print(f"Max length: {max_length_filtered}, at index: {max_index_filtered}")
    print(print(filtered.loc[max_index_filtered]))
    print()
    
    print(filtered["length"].min())
    print()

    diff_mask = df["gene_name_x"] != df["gene_name_y"]
    any_diff = diff_mask.any()
    print("Any gene names differ:", any_diff)
    print()

    print((df["total_counts"] == 0).sum())
