#yeast specific data 
#read in two files
# filter out where quality = activator from the first file save it as file_remove_activator_p.csv
#map "sequence" column of file_remove_activator_p.csv to the second file data column "Fragment Sequence" and filter out the ones where PADI Score is not greater than 1 --> save that file to y_activators.csv
#combine that data in the filterToFinalData file
#use that data

import pandas as pd
#yeast_specificdata.py

file1 = pd.read_csv("/Users/anyalachwani/Downloads/all_data_6_2026_relaxed_annotated_features.csv")          # plant dataset
file2 = pd.read_excel("/Users/anyalachwani/Downloads/Morffy_all_data.xlsx")       # Morffy dataset

morffy_cleaned = file2[
    file2["PADI Score"].notna()
].copy()

morffy_cleaned.to_excel(
    "morffy_cleaned.xlsx",
    index=False
)

# Remove activators from file1


file_remove_activator = file1[
    (file1["quality"].str.strip() != "activator") &
    (file1["quality"].str.strip() != "ambiguous_activator")
].copy()


file_remove_activator.to_csv(
    "file_remove_activator_p.csv",
    index=False
)

print("Rows after removing activators and ambiguous_activator",
      len(file_remove_activator))


# Keep only sequences that appear in Morffy
# with PADI Score > 1

morffy_positive = morffy_cleaned[
    morffy_cleaned["PADI Score"] > 1
].copy()

# Get all valid Fragment Sequences
positive_sequences = set(
    morffy_positive["Fragment Sequence"]
    .astype(str)
    .str.strip()
)

# Keep only rows whose Sequence exists in Fragment Sequence
y_activators = file_remove_activator[
    file_remove_activator["protein_seq"]
    .astype(str)
    .str.strip()
    .isin(positive_sequences)
].copy()

y_activators.to_csv("y_activators.csv", index=False)

print("Rows in y_activators:", len(y_activators))

