import os
import sys
from pathlib import Path
import spyrmsd
from spyrmsd import rmsd
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import numpy as np

def compute_symmetry_corrected_rmsds_with_rdkit(docked_path_str: str, reference_path_str: str):
    """
    Computes the symmetry-corrected RMSD for each molecule in a docked SDF file
    against a reference molecule using RDKit to load molecules and spyrmsd for RMSD calculation.
    Removes hydrogen atoms before RMSD calculation.

    Args:
        docked_path_str (str): Path to the SDF file containing docked poses.
        reference_path_str (str): Path to the SDF file containing the reference structure.

    Returns:
        list: A list of RMSD values corresponding to each molecule in the docked file.
              Returns None if an error occurs during processing.
    """
    try:
        # Load the reference molecule using RDKit
        ref_supplier = Chem.SDMolSupplier(reference_path_str, removeHs=False) # Keep Hs initially to let spyrmsd handle stripping
        ref_mols = []
        for mol in ref_supplier:
            if mol is not None:
                ref_mols.append(mol)
                # Only use the first valid molecule from the reference file
                break 
        if not ref_mols:
            print(f"Error: No valid molecules found in the reference file '{reference_path_str}'.")
            return None
        
        ref_mol = ref_mols[0]
        print(f"Loaded reference molecule: {ref_mol.GetProp('_Name') if ref_mol.HasProp('_Name') else 'Unknown'}")

        # Convert the reference molecule to spyrmsd format using the from_rdkit constructor
        from spyrmsd.molecule import Molecule
        ref_spy_mol = Molecule.from_rdkit(ref_mol)
        if ref_spy_mol is None:
            print("Error: Could not convert reference molecule to spyrmsd format using from_rdkit.")
            return None

        # Strip Hydrogens from the reference molecule using spyrmsd's built-in function
        ref_spy_mol.strip()
        print(f"Stripped Hs from reference. Remaining atoms: {ref_spy_mol.natoms}")

        # Load all docked molecules using RDKit
        docked_supplier = Chem.SDMolSupplier(docked_path_str, removeHs=False) # Keep Hs initially to let spyrmsd handle stripping
        docked_mols = []
        for mol in docked_supplier:
            if mol is not None:
                docked_mols.append(mol)
        if not docked_mols:
            print(f"Error: No valid molecules found in the docked file '{docked_path_str}'.")
            return None
        
        print(f"Loaded {len(docked_mols)} docked poses.")

        # Extract reference properties from the spyrmsd object (after stripping Hs)
        ref_coords = ref_spy_mol.coordinates
        ref_atomicnums = ref_spy_mol.atomicnums
        ref_adjacency_matrix = ref_spy_mol.adjacency_matrix

        # Initialize a list to store the calculated RMSD values
        rmsd_values = []

        # Iterate through each docked pose (molecule) in the docked file
        for i, docked_rdkit_mol in enumerate(docked_mols):
            try:
                # Convert the docked molecule to spyrmsd format using the from_rdkit constructor
                docked_spy_mol = Molecule.from_rdkit(docked_rdkit_mol)
                if docked_spy_mol is None:
                    print(f"Warning: Skipping pose {i+1}, could not convert to spyrmsd format using from_rdkit.")
                    rmsd_values.append(float('nan'))
                    continue

                # Strip Hydrogens from the docked molecule using spyrmsd's built-in function
                docked_spy_mol.strip()
                print(f"  Processing Pose {i+1} ({docked_rdkit_mol.GetProp('_Name') if docked_rdkit_mol.HasProp('_Name') else f'Pose_{i+1}'}). Stripped Hs. Remaining atoms: {docked_spy_mol.natoms}")

                # Extract docked pose properties from the spyrmsd object (after stripping Hs)
                docked_coords = docked_spy_mol.coordinates
                docked_atomicnums = docked_spy_mol.atomicnums
                docked_adjacency_matrix = docked_spy_mol.adjacency_matrix

                # Check if the number of atoms matches after stripping
                if ref_coords.shape != docked_coords.shape or \
                   ref_atomicnums.shape != docked_atomicnums.shape or \
                   ref_adjacency_matrix.shape != docked_adjacency_matrix.shape:
                    print(f"    Error: Atom count mismatch after stripping Hs.")
                    print(f"      Ref: coords {ref_coords.shape}, atomicnums {ref_atomicnums.shape}, adj {ref_adjacency_matrix.shape}")
                    print(f"      Docked: coords {docked_coords.shape}, atomicnums {docked_atomicnums.shape}, adj {docked_adjacency_matrix.shape}")
                    rmsd_values.append(float('nan'))
                    continue


                # Compute the symmetry-corrected RMSD using the H-stripped data
                sc_rmsd = rmsd.symmrmsd(
                    ref_coords,          # Coordinates of the H-stripped reference molecule
                    docked_coords,       # Coordinates of the H-stripped docked molecule
                    ref_atomicnums,      # Atomic numbers for H-stripped reference molecule
                    docked_atomicnums,   # Atomic numbers for H-stripped docked molecule
                    ref_adjacency_matrix, # Adjacency matrix for the H-stripped reference molecule
                    docked_adjacency_matrix, # Adjacency matrix for the H-stripped docked molecule
                )
                # Append the calculated RMSD to the list
                rmsd_values.append(sc_rmsd)
                mol_name = docked_rdkit_mol.GetProp('_Name') if docked_rdkit_mol.HasProp('_Name') else f'Pose_{i+1}'
                print(f"Pose {i+1} ({mol_name}): Symmetry-corrected RMSD = {sc_rmsd:.4f} Å")

            except Exception as e:
                # Handle potential errors during RMSD calculation for individual poses
                mol_name = docked_rdkit_mol.GetProp('_Name') if docked_rdkit_mol.HasProp('_Name') else f'Pose_{i+1}'
                print(f"Error calculating RMSD for pose {i+1} ({mol_name}): {e}")
                # Print the traceback for the specific error
                import traceback
                traceback.print_exc()
                # Append NaN to maintain list indexing alignment if needed, allowing other poses to be processed
                rmsd_values.append(float('nan'))

        return rmsd_values

    except Exception as e:
        # Handle general errors during file loading or processing
        print(f"An unexpected error occurred while processing the SDF files: {e}")
        import traceback
        traceback.print_exc() # Print the full traceback for debugging
        return None

def get_pdb_id_from_dir_name(dir_path):
    """Extract PDB ID from directory name (first 4 characters)"""
    dir_name = dir_path.name  # Use .name instead of rstrip('/')
    if len(dir_name) >= 4:
        return dir_name[:4]
    else:
        return None

def get_affinities_from_sdf(sdf_path):
    """
    Extract minimizedAffinity values from SDF file
    """
    affinities = []
    with open(sdf_path, 'r') as f:
        content = f.read()
    
    # Split by '$$$$' to separate molecules
    molecules = content.split('$$$$')
    
    for mol_block in molecules:
        lines = mol_block.split('\n')
        for line in lines:
            if '<minimizedAffinity>' in line:
                # Get the next line which contains the affinity value
                idx = lines.index(line)
                if idx + 1 < len(lines):
                    try:
                        affinity = float(lines[idx + 1].strip())
                        affinities.append(affinity)
                    except ValueError:
                        print(f"Could not parse affinity: {lines[idx + 1].strip()}")
                break
    
    return affinities

def process_single_directory(dir_path):
    """Process a single directory to compute RMSD from existing exhaust50_dock.sdf"""
    dir_path = Path(dir_path)
    full_dir_name = dir_path.name  # e.g., "7Q5I_I0F"
    
    # Look for the output file and reference ligand
    output_file = dir_path / "exhaust50_dock.sdf"
    ligand_autobox_file = dir_path / f"{full_dir_name}_ligand.sdf"
    
    # Debug: Print the actual file paths and check if they exist
    print(f"Checking files in {dir_path}:")
    print(f"  Full directory name: {full_dir_name}")
    print(f"  Output file: {output_file} - exists: {output_file.exists()}")
    print(f"  Ligand autobox file: {ligand_autobox_file} - exists: {ligand_autobox_file.exists()}")
    
    # Check if required files exist
    if not (output_file.exists() and ligand_autobox_file.exists()):
        print(f"Warning: Required files not found in {dir_path}, skipping...")
        return None
    
    print(f"Processing directory: {dir_path}")
    print(f"Computing symmetry-corrected RMSDs for {full_dir_name}...")
    
    # Extract affinities from the docked file
    affinities = get_affinities_from_sdf(str(output_file))
    print(f"Found {len(affinities)} affinities in {output_file}")
    
    # Compute symmetry-corrected RMSDs
    rmsd_values = compute_symmetry_corrected_rmsds_with_rdkit(str(output_file), str(ligand_autobox_file))
    
    if rmsd_values is not None:
        # Filter out NaN values
        valid_rmsds = [r for r in rmsd_values if r is not None and not (isinstance(r, float) and np.isnan(r))]
        
        if valid_rmsds and len(affinities) > 0:
            # Sort RMSD values to get top poses
            sorted_rmsds = sorted(valid_rmsds)
            
            # Get top 1 RMSD (lowest RMSD values)
            top1_rmsd = sorted_rmsds[0] if len(sorted_rmsds) > 0 else None
            
            # Calculate percentage of RMSDs < 2 Å (valid poses)
            rmsds_less_than_2 = [r for r in valid_rmsds if r < 2.0]
            percentage_less_than_2 = (len(rmsds_less_than_2) / len(valid_rmsds)) * 100 if valid_rmsds else 0
            
            # Sort affinities (original order based on energy)
            sorted_affinities = sorted(affinities)
            
            # Get the RMSD of the pose with the best (lowest) affinity
            best_affinity_idx = affinities.index(min(affinities))
            best_affinity_rmsd = rmsd_values[best_affinity_idx] if best_affinity_idx < len(rmsd_values) else None
            
            # Find pose numbers (sorted by RMSD, lowest first) that have RMSD < 2
            rmsd_indices = [(i, rmsd_values[i]) for i in range(len(rmsd_values))]
            sorted_by_rmsd = sorted(rmsd_indices, key=lambda x: x[1])  # Sort by RMSD value
            
            poses_with_rmsd_less_than_2_by_rmsd = []
            for idx, (original_idx, rmsd_val) in enumerate(sorted_by_rmsd):
                if rmsd_val < 2.0:
                    # Find the pose rank when sorted by affinity
                    pose_rank_by_affinity = sorted(range(len(affinities)), key=lambda i: affinities[i]).index(original_idx) + 1
                    poses_with_rmsd_less_than_2_by_rmsd.append(pose_rank_by_affinity)
            
            # Find the pose with the lowest RMSD and its rank/affinity
            lowest_rmsd_idx = valid_rmsds.index(top1_rmsd) if top1_rmsd is not None else -1
            lowest_rmsd_pose_rank = sorted_affinity_indices = sorted(range(len(affinities)), key=lambda i: affinities[i]).index(lowest_rmsd_idx) + 1 if lowest_rmsd_idx != -1 else -1
            lowest_rmsd_affinity = affinities[lowest_rmsd_idx] if lowest_rmsd_idx != -1 else None
            
            # Check if the best affinity pose is also the lowest RMSD pose
            is_best_affinity_lowest_rmsd = (best_affinity_rmsd == top1_rmsd)
            
            # Get affinities for poses with RMSD < 2 (sorted by RMSD)
            affinities_for_valid_poses_by_rmsd = []
            for idx, (original_idx, rmsd_val) in enumerate(sorted_by_rmsd):
                if rmsd_val < 2.0:
                    # Find the pose rank when sorted by affinity
                    pose_rank_by_affinity = sorted(range(len(affinities)), key=lambda i: affinities[i]).index(original_idx) + 1
                    affinities_for_valid_poses_by_rmsd.append((pose_rank_by_affinity, affinities[original_idx]))
            
            print(f"PDB ID {full_dir_name}:")
            print(f"  Lowest RMSD: {top1_rmsd:.4f} Å")
            print(f"  Pose rank of lowest RMSD: {lowest_rmsd_pose_rank}")
            print(f"  Affinity of lowest RMSD pose: {lowest_rmsd_affinity:.3f}")
            print(f"  Best affinity pose RMSD > 2 Å: {best_affinity_rmsd > 2.0 if best_affinity_rmsd is not None else 'N/A'}")
            print(f"  Best affinity pose is lowest RMSD: {is_best_affinity_lowest_rmsd}")
            print(f"  Poses with RMSD < 2 Å (sorted by RMSD): {poses_with_rmsd_less_than_2_by_rmsd}")
            print(f"  Affinities for poses with RMSD < 2 Å (sorted by RMSD): {[(p, f'{a:.3f}') for p, a in affinities_for_valid_poses_by_rmsd]}")
            
            return {
                'pdb_id': full_dir_name,
                'top1_rmsd': top1_rmsd,  # Lowest RMSD among all poses
                'best_affinity_rmsd': best_affinity_rmsd,  # RMSD of the pose with best affinity
                'percentage_less_than_2': percentage_less_than_2,
                'total_poses': len(valid_rmsds),
                'valid_poses': len(rmsds_less_than_2),
                'affinities': affinities,
                'sorted_affinities': sorted_affinities,
                'poses_with_rmsd_less_than_2_by_rmsd': poses_with_rmsd_less_than_2_by_rmsd,
                'lowest_rmsd_pose_rank': lowest_rmsd_pose_rank,
                'lowest_rmsd_affinity': lowest_rmsd_affinity,
                'affinities_for_valid_poses_by_rmsd': affinities_for_valid_poses_by_rmsd,
                'is_best_affinity_lowest_rmsd': is_best_affinity_lowest_rmsd
            }
        else:
            print(f"No valid RMSD values or affinities computed for {full_dir_name}")
            return None
    else:
        print(f"Failed to compute RMSDs for {full_dir_name}")
        return None

def main():
    # Get all directories at depth=1 from current directory
    current_dir = Path('.')
    depth1_dirs = [d for d in current_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(depth1_dirs)} directories to process")
    
    # Process directories sequentially (single CPU) - only compute RMSD from existing files
    results = []
    for dir_path in depth1_dirs:
        result = process_single_directory(dir_path)
        if result is not None:
            results.append(result)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    for result in results:
        print(f"PDB ID: {result['pdb_id']}")
        print(f"  Lowest RMSD: {result['top1_rmsd']:.4f} Å")
        print(f"  Pose rank of lowest RMSD: {result['lowest_rmsd_pose_rank']}")
        print(f"  Affinity of lowest RMSD pose: {result['lowest_rmsd_affinity']:.3f}")
        print(f"  Best affinity pose RMSD > 2 Å: {result['best_affinity_rmsd'] > 2.0 if result['best_affinity_rmsd'] is not None else 'N/A'}")
        print(f"  Best affinity pose is lowest RMSD: {result['is_best_affinity_lowest_rmsd']}")
        print(f"  Poses with RMSD < 2 Å (sorted by RMSD): {result['poses_with_rmsd_less_than_2_by_rmsd']}")
        print(f"  Affinities for poses with RMSD < 2 Å (sorted by RMSD): {[(p, f'{a:.3f}') for p, a in result['affinities_for_valid_poses_by_rmsd']]}")
        print("-" * 50)
    
    # Calculate statistics
    best_affinity_rmsd_over_2 = sum(1 for r in results if r['best_affinity_rmsd'] is not None and r['best_affinity_rmsd'] > 2.0)
    not_lowest_rmsd = sum(1 for r in results if r['is_best_affinity_lowest_rmsd'] == False)
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total systems processed: {len(results)}")
    print(f"Number of systems where best affinity pose RMSD > 2 Å: {best_affinity_rmsd_over_2}")
    print(f"Number of systems where best affinity pose is NOT the lowest RMSD: {not_lowest_rmsd}")
    print(f"Percentage of systems where best affinity pose RMSD > 2 Å: {best_affinity_rmsd_over_2/len(results)*100:.2f}%")
    print(f"Percentage of systems where best affinity pose is NOT the lowest RMSD: {not_lowest_rmsd/len(results)*100:.2f}%")
    
    # Save results to a single text file
    with open("exhaust50_results_summary_with_affinities.txt", "w") as f:
        f.write("EXHAUST50 DOCKING RESULTS SUMMARY WITH AFFINITIES\n")
        f.write("="*80 + "\n")
        f.write(f"Total directories processed: {len(results)}\n")
        f.write(f"Directories with successful results: {len([r for r in results if r is not None])}\n")
        f.write("="*80 + "\n")
        
        for result in results:
            f.write(f"PDB ID: {result['pdb_id']}\n")
            f.write(f"  Lowest RMSD: {result['top1_rmsd']:.4f} Å\n")
            f.write(f"  Pose rank of lowest RMSD: {result['lowest_rmsd_pose_rank']}\n")
            f.write(f"  Affinity of lowest RMSD pose: {result['lowest_rmsd_affinity']:.3f}\n")
            f.write(f"  Best affinity pose RMSD > 2 Å: {result['best_affinity_rmsd'] > 2.0 if result['best_affinity_rmsd'] is not None else 'N/A'}\n")
            f.write(f"  Best affinity pose is lowest RMSD: {result['is_best_affinity_lowest_rmsd']}\n")
            f.write(f"  Poses with RMSD < 2 Å (sorted by RMSD): {result['poses_with_rmsd_less_than_2_by_rmsd']}\n")
            f.write(f"  Affinities for poses with RMSD < 2 Å (sorted by RMSD): {[(p, f'{a:.3f}') for p, a in result['affinities_for_valid_poses_by_rmsd']]}\n")
            f.write("-" * 50 + "\n")
        
        # Add overall statistics
        f.write(f"\nOVERALL STATISTICS\n")
        f.write(f"Total systems processed: {len(results)}\n")
        f.write(f"Number of systems where best affinity pose RMSD > 2 Å: {best_affinity_rmsd_over_2}\n")
        f.write(f"Number of systems where best affinity pose is NOT the lowest RMSD: {not_lowest_rmsd}\n")
        f.write(f"Percentage of systems where best affinity pose RMSD > 2 Å: {best_affinity_rmsd_over_2/len(results)*100:.2f}%\n")
        f.write(f"Percentage of systems where best affinity pose is NOT the lowest RMSD: {not_lowest_rmsd/len(results)*100:.2f}%\n")

if __name__ == "__main__":
    main()
