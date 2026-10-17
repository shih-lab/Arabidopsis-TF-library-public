import pandas as pd
#description: getting plants only activators compared against yeast file in order to do plants only vs. shared activator comparison 
#updated plantsonlyactivatorsNOTyeast.py
activators = pd.read_csv("/Users/anyalachwani/Downloads/activators_only_plantsdata_finaldata_june5.csv")

activators = activators.drop_duplicates(
    subset="#Feature",
    keep="first"
)
fragments = pd.read_excel("/Users/anyalachwani/Downloads/Morffy_all_data.xlsx")

# Create sets
feature_set = set(
    activators["#Feature"].astype(str).str.strip()
)

morffy_set = set(
    fragments["Fragment Name"].astype(str).str.strip()
)


print("Activators after removing non-Morffy features:", len(activators))
print("Unique activators remaining:",
      activators["#Feature"].nunique())

print("activators_length", len(activators))

activators.to_csv(
    "/Users/anyalachwani/Downloads/activators_only_plantsdata_finaldata_june5_cleaned.csv",
    index=False
)



feature_set = set(
    activators["#Feature"].astype(str).str.strip()
)

morffy_set = set(
    fragments["Fragment Name"].astype(str).str.strip()
)

print("Unique activators:", len(feature_set))
print("Unique Morffy names:", len(morffy_set))

missing = feature_set - morffy_set

print("\nFeatures present in activators but missing from Morffy:", len(missing))

for x in sorted(missing):
    print(x)

matched = fragments[
    fragments["Fragment Name"].astype(str).str.strip().isin(feature_set)
]



print("Matched rows:", len(matched))
print("Matched unique names:", matched["Fragment Name"].nunique())


nan_padi = fragments[
    (fragments["Fragment Name"].astype(str).str.strip().isin(feature_set))
    & (fragments["PADI Score"].isna())
]

print("\nMatched activators with NaN PADI:", len(nan_padi))

print(
    nan_padi[["Fragment Name", "PADI Score"]]
)

activators.columns = activators.columns.str.strip()
fragments.columns = fragments.columns.str.strip()


feature_set = set(activators["#Feature"].astype(str))


#filtered_fragments = fragments[
 #   (fragments["Fragment Name"].astype(str).isin(feature_set)) &
  #  (fragments["PADI Score"] <= 1)
#]


#filtered_fragments.to_csv("plantsOnlyActivatorsNOTyeast_finalData_june5.csv", index=False)

#print("Total fragments:", len(fragments))
#print("Matched rows with PADI < 1:", len(filtered_fragments))


padi_le1_names = set(
    fragments.loc[
        fragments["PADI Score"] <= 1,
        "Fragment Name"
    ].astype(str).str.strip()
)

plant_only_activators = activators[
    activators["#Feature"]
        .astype(str)
        .str.strip()
        .isin(padi_le1_names)
]

plant_only_activators.to_csv(
    "/Users/anyalachwani/Downloads/plantsOnlyActivatorsNOTyeast_finalData_june5.csv",
    index=False
)

print("Plant-only activator rows:",
      len(plant_only_activators))
print("Plant-only unique activators:",
      plant_only_activators["#Feature"].nunique())

