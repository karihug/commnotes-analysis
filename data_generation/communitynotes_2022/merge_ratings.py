import os

import pandas as pd


# Path to folder containing ratings files
ratings_folder = "ratings2023"

# Output file name
output_file = "merged_ratings2023.tsv"

# Initialize an empty DataFrame to store merged ratings
merged_ratings = pd.DataFrame()

# Iterate over files in the ratings folder
for filename in os.listdir(ratings_folder):
    if filename.endswith(".tsv"):
        file_path = os.path.join(ratings_folder, filename)
        ratings = pd.read_csv(file_path, sep="\t")
        merged_ratings = pd.concat([merged_ratings, ratings], ignore_index=True)

# Save the merged DataFrame to a single TSV file
merged_ratings.to_csv(output_file, sep="\t", index=False)

print(f"Merged ratings saved to {output_file}")
