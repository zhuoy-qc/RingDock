import pandas as pd
import re

def analyze_pi_cation_recovery_detailed():
    # Read the CSV files
    sampled_df = pd.read_csv('all_sampled_poses_with-pi-cation-interactions.csv')
    ref_df = pd.read_csv('ref-pi-cation-interactions.csv')

    # Group reference interactions by PDB ID extracted from Directory column
    ref_by_pdb = {}
    for _, row in ref_df.iterrows():
        # Extract PDB ID from Directory column
        directory = row['Directory']
        # For format like: 8A2D_KXY, extract the part before underscore if exists
        if '_' in directory:
            pdb_id = directory
        else:
            pdb_id = directory

        if pdb_id not in ref_by_pdb:
            ref_by_pdb[pdb_id] = []
        ref_by_pdb[pdb_id].append((row['Protein'], row['Ring_Type']))

    # Get pose 1 interactions grouped by PDB ID
    pose1_by_pdb = {}
    for _, row in sampled_df.iterrows():
        # Extract the pose number from PDB_File column
        pdb_file = row['PDB_File']
        match = re.search(r'complex_(\d+)\.pdb', pdb_file)
        if match and int(match.group(1)) == 1:
            # Extract PDB ID from the PDB_File (before the pose part)
            # For format like: 6QU4_JJ2_exhaust50_complex_25.pdb
            base_parts = pdb_file.split('_')
            # Remove the last two parts which are 'complex_XX.pdb'
            extracted_pdb_id = '_'.join(base_parts[:-2])

            # If this doesn't work well, use PDB_ID column if it exists
            if hasattr(row, 'PDB_ID') and pd.notna(row['PDB_ID']):
                pdb_id = row['PDB_ID']
            elif extracted_pdb_id == '' or extracted_pdb_id == pdb_file.split('_')[0]:
                # If extraction didn't work properly, use directory from the row itself
                # This assumes sampled_df might have a Directory column too
                if 'Directory' in sampled_df.columns:
                    dir_val = row['Directory']
                    if '_' in dir_val:
                        pdb_id = dir_val
                    else:
                        pdb_id = dir_val
                else:
                    # Last resort: use the extracted ID even if imperfect
                    pdb_id = extracted_pdb_id
            else:
                pdb_id = extracted_pdb_id

            if pdb_id not in pose1_by_pdb:
                pose1_by_pdb[pdb_id] = []
            pose1_by_pdb[pdb_id].append((row['Protein'], row['Ring_Type']))

    print("Detailed Pose 1 Recovery Analysis by PDB ID:")
    print("="*60)

    total_correct_recoveries = 0
    total_false_positives = 0
    total_misses = 0

    # Analyze each PDB ID
    for pdb_id in sorted(ref_by_pdb.keys()):
        ref_interactions = set(ref_by_pdb[pdb_id])
        pose1_interactions = set(pose1_by_pdb.get(pdb_id, []))

        print(f"\nPDB ID: {pdb_id}")
        print(f"  Reference interactions: {len(ref_interactions)}")
        for prot, ring_type in sorted(ref_interactions):
            print(f"    - Protein: {prot}, Ring Type: {ring_type}")

        if pdb_id in pose1_by_pdb:
            print(f"  Pose 1 predictions: {len(pose1_interactions)}")
            for prot, ring_type in sorted(pose1_interactions):
                print(f"    - Protein: {prot}, Ring Type: {ring_type}")
        else:
            print(f"  Pose 1 predictions: 0 (no pose 1 data found)")

        # Calculate correct recoveries, false positives, and misses
        correct_recoveries = len(ref_interactions.intersection(pose1_interactions))
        false_positives = len(pose1_interactions) - correct_recoveries if pdb_id in pose1_by_pdb else 0
        misses = len(ref_interactions) - correct_recoveries

        total_correct_recoveries += correct_recoveries
        total_false_positives += false_positives
        total_misses += misses

        print(f"  Correctly recovered: {correct_recoveries}")
        print(f"  False positives: {false_positives}")
        print(f"  Misses: {misses}")

        # Show details if there are discrepancies
        if correct_recoveries < len(ref_interactions) or false_positives > 0:
            print("  Details:")
            if ref_interactions.intersection(pose1_interactions):
                print(f"    Correctly recovered: {sorted(ref_interactions.intersection(pose1_interactions))}")
            if pose1_interactions - ref_interactions:
                print(f"    False positives: {sorted(pose1_interactions - ref_interactions)}")
            if ref_interactions - pose1_interactions:
                                print(f"    Missed: {sorted(ref_interactions - pose1_interactions)}")

    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Total correct recoveries: {total_correct_recoveries}")
    print(f"Total false positives: {total_false_positives}")
    print(f"Total misses: {total_misses}")

    total_reference_interactions = sum(len(interactions) for interactions in ref_by_pdb.values())
    if total_reference_interactions > 0:
        print(f"Total ref interactions: {total_reference_interactions}")
        print(f"Overall recovery rate: {(total_correct_recoveries / total_reference_interactions) * 100:.2f}%")

# Run the analysis
analyze_pi_cation_recovery_detailed()
