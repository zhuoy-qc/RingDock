import pandas as pd
import numpy as np

def find_matches_for_target_values(file_path, target_distance=4.2, target_offset=0.0, tolerance=0.1):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Find rows that match the criteria with tolerance
    distance_condition = np.abs(df['distance'] - target_distance) <= tolerance
    offset_condition = np.abs(df['offset'] - target_offset) <= tolerance
    
    # Get matching rows
    matches = df[distance_condition & offset_condition]
    
    print(f"Finding matches for distance ≈ {target_distance} (±{tolerance}) and offset ≈ {target_offset} (±{tolerance})")
    print("-" * 80)
    
    if len(matches) > 0:
        print(f"Total matches found: {len(matches)}")
        print(f"Matched dihedral values: {sorted(matches['dihedral'].tolist())}")
        
        # Print detailed information about matches
        print("\nDetailed matches:")
        for idx, row in matches.iterrows():
            print(f"  Distance: {row['distance']:.3f}, Offset: {row['offset']:.3f}, "
                  f"RZ: {row['rz']:.3f}, Dihedral: {row['dihedral']:.3f}, Energy: {row['energy']:.3f}")
    else:
        print("No matches found.")
    
    return matches

# Usage
if __name__ == "__main__":
    file_path = "combined_sorted.csv"
    matches = find_matches_for_target_values(file_path, target_distance=4.2, target_offset=0.0, tolerance=0.1)
