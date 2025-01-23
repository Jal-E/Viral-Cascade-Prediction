import csv

# Path to your CSV file
csv_file = 'cascade_features.csv'

positive_count = 0
non_positive_count = 0  # For 0 or negative values

with open(csv_file, 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert the virality value to a float or int
        virality_str = row['virality'].strip()
        
        # If there's a possibility of empty entries, handle that gracefully
        if virality_str == '':
            # If an empty virality should count as non-positive, do that here
            non_positive_count += 1
            continue
        
        virality = float(virality_str)
        
        if virality > 0:
            positive_count += 1
        else:
            non_positive_count += 1

print("Number of positives (virality = 1):", positive_count)
print("Number of non-positives (virality = 0):", non_positive_count)
