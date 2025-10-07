import pandas as pd
import numpy as np

df = pd.read_csv('object_positions.csv')

# Find duplicate rows where all coordinates are within 5 pixels of another row
def find_duplicate_rows(df, threshold=20):
    duplicate_indices = set()
    
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            # Check if all 4 coordinates are within threshold
            cup_x_close = abs(df.iloc[i]['cup_x'] - df.iloc[j]['cup_x']) <= threshold
            cup_y_close = abs(df.iloc[i]['cup_y'] - df.iloc[j]['cup_y']) <= threshold
            block_x_close = abs(df.iloc[i]['block_x'] - df.iloc[j]['block_x']) <= threshold
            block_y_close = abs(df.iloc[i]['block_y'] - df.iloc[j]['block_y']) <= threshold
            
            if cup_x_close and cup_y_close and block_x_close and block_y_close:
                # Mark the later row (j) as duplicate
                duplicate_indices.add(j)
    
    return list(duplicate_indices)
qqwq
# Find duplicate row indices
duplicate_indices = find_duplicate_rows(df, threshold=100)
close_rows = df.iloc[duplicate_indices]

# Display the information
print(f"\n{'='*60}")
print(f"Total rows in dataset: {len(df)}")
print(f"Rows where cup and block are within 5 pixels (duplicates): {len(close_rows)}")
print(f"{'='*60}\n")

if len(close_rows) > 0:
    print("Rows that would be deleted:")
    print(close_rows)
    print(f"\n{'='*60}\n")
    
    # Ask for confirmation
    response = input("Do you want to delete these rows? (yes/no): ").strip().lower()
    
    if response == 'yes':
        # Delete the rows
        df = df.drop(duplicate_indices).reset_index(drop=True)
        # Save back to CSV
        df.to_csv('object_positions.csv', index=False)
        print(f"\n✓ Deleted {len(close_rows)} rows. CSV file updated.")
        print(f"Remaining rows: {len(df)}")
    else:
        print("\n✗ No rows deleted. CSV file unchanged.")
else:
    print("No rows found where cup and block positions are within 5 pixels of each other.")