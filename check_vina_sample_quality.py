import os
import sys
from pathlib import Path
import spyrmsd
from spyrmsd import rmsd
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import numpy as np

def compute_symmetry_corrected_rmsds_with_rdkit(docked_path_str: str, reference_path_str: str):
    try:
        ref_supplier = Chem.SDMolSupplier(reference_path_str, removeHs=False)
        ref_mols = []
        for mol in ref_supplier:
            if mol is not None:
                ref_mols.append(mol)
                break 
        if not ref_mols:
            print(f"Error: No valid molecules found in the reference file '{reference_path_str}'.")
            return None
        
        ref_mol = ref_mols[0]
        print(f"Loaded reference molecule: {ref_mol.GetProp('_Name') if ref_mol.HasProp('_Name') else 'Unknown'}")

        from spyrmsd.molecule import Molecule
        ref_spy_mol = Molecule.from_rdkit(ref_mol)
        if ref_spy_mol is None:
            print("Error: Could not convert reference molecule to spyrmsd format using from_rdkit.")
            return None

        ref_spy_mol.strip()
        print(f"Stripped Hs from reference. Remaining atoms: {ref_spy_mol.natoms}")

        docked_supplier = Chem.SDMolSupplier(docked_path_str, removeHs=False)
        docked_mols = []
        for mol in docked_supplier:
            if mol is not None:
                docked_mols.append(mol)
        if not docked_mols:
            print(f"Error: No valid molecules found in the docked file '{docked_path_str}'.")
            return None
        
        print(f"Loaded {len(docked_mols)} docked poses.")

        ref_coords = ref_spy_mol.coordinates
        ref_atomicnums = ref_spy_mol.atomicnums
        ref_adjacency_matrix = ref_spy_mol.adjacency_matrix

        rmsd_values = []

        for i, docked_rdkit_mol in enumerate(docked_mols):
            try:
                docked_spy_mol = Molecule.from_rdkit(docked_rdkit_mol)
                if docked_spy_mol is None:
                    print(f"Warning: Skipping pose {i+1}, could not convert to spyrmsd format using from_rdkit.")
                    rmsd_values.append(float('nan'))
                    continue

                docked_spy_mol.strip()
                print(f"  Processing Pose {i+1} ({docked_rdkit_mol.GetProp('_Name') if docked_rdkit_mol.HasProp('_Name') else f'Pose_{i+1}'}). Stripped Hs. Remaining atoms: {docked_spy_mol.natoms}")

                docked_coords = docked_spy_mol.coordinates
                docked_atomicnums = docked_spy_mol.atomicnums
                docked_adjacency_matrix = docked_spy_mol.adjacency_matrix

                if ref_coords.shape != docked_coords.shape or \
                   ref_atomicnums.shape != docked_atomicnums.shape or \
                   ref_adjacency_matrix.shape != docked_adjacency_matrix.shape:
                    print(f"    Error: Atom count mismatch after stripping Hs.")
                    print(f"      Ref: coords {ref_coords.shape}, atomicnums {ref_atomicnums.shape}, adj {ref_adjacency_matrix.shape}")
                    print(f"      Docked: coords {docked_coords.shape}, atomicnums {docked_atomicnums.shape}, adj {docked_adjacency_matrix.shape}")
                    rmsd_values.append(float('nan'))
                    continue

                sc_rmsd = rmsd.symmrmsd(
                    ref_coords,          
                    docked_coords,       
                    ref_atomicnums,      
                    docked_atomicnums,   
                    ref_adjacency_matrix, 
                    docked_adjacency_matrix, 
                )
                rmsd_values.append(sc_rmsd)
                mol_name = docked_rdkit_mol.GetProp('_Name') if docked_rdkit_mol.HasProp('_Name') else f'Pose_{i+1}'
                print(f"Pose {i+1} ({mol_name}): Symmetry-corrected RMSD = {sc_rmsd:.4f} Å")

            except Exception as e:
                mol_name = docked_rdkit_mol.GetProp('_Name') if docked_rdkit_mol.HasProp('_Name') else f'Pose_{i+1}'
                print(f"Error calculating RMSD for pose {i+1} ({mol_name}): {e}")
                import traceback
                traceback.print_exc()
                rmsd_values.append(float('nan'))

        return rmsd_values

    except Exception as e:
        print(f"An unexpected error occurred while processing the SDF files: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_pdb_id_from_dir_name(dir_path):
    dir_name = dir_path.name
    if len(dir_name) >= 4:
        return dir_name[:4]
    else:
        return None

def get_affinities_from_sdf(sdf_path):
    affinities = []
    with open(sdf_path, 'r') as f:
        content = f.read()
    
    molecules = content.split('$$$$')
    
    for mol_block in molecules:
        lines = mol_block.split('\n')
        for line in lines:
            if '<minimizedAffinity>' in line:
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
    dir_path = Path(dir_path)
    full_dir_name = dir_path.name

    output_file = dir_path / "exhaust50_dock.sdf"
    ligand_autobox_file = dir_path / f"{full_dir_name}_ligand.sdf"

    print(f"Checking files in {dir_path}:")
    print(f"  Full directory name: {full_dir_name}")
    print(f"  Output file: {output_file} - exists: {output_file.exists()}")
    print(f"  Ligand autobox file: {ligand_autobox_file} - exists: {ligand_autobox_file.exists()}")

    if not (output_file.exists() and ligand_autobox_file.exists()):
        print(f"Warning: Required files not found in {dir_path}, skipping...")
        return None

    print(f"Processing directory: {dir_path}")
    print(f"Computing symmetry-corrected RMSDs for {full_dir_name}...")

    affinities = get_affinities_from_sdf(str(output_file))
    print(f"Found {len(affinities)} affinities in {output_file}")

    rmsd_values = compute_symmetry_corrected_rmsds_with_rdkit(str(output_file), str(ligand_autobox_file))

    if rmsd_values is not None:
        # Calculate Vina ranks
        vina_ranks = get_vina_ranks(affinities)

        # Create detailed results for each pose
        detailed_results = []
        for i, (rmsd_val, affinity, rank) in enumerate(zip(rmsd_values, affinities, vina_ranks)):
            if not (isinstance(rmsd_val, float) and np.isnan(rmsd_val)):
                pdb_id = full_dir_name
                pose_num = i + 1
                detailed_results.append({
                    'PDB_ID': pdb_id,
                    'Pose': pose_num,
                    'RMSD': round(rmsd_val, 4),
                    'Vina_Score': round(affinity, 3),
                    'Vina_Rank': rank
                })

        if detailed_results:
            print(f"Generated {len(detailed_results)} pose entries for {full_dir_name}")
            return detailed_results
        else:
            print(f"No valid results generated for {full_dir_name}")
            return None
    else:
        print(f"Failed to compute RMSDs for {full_dir_name}")
        return None

def main():
    current_dir = Path('.')
    depth1_dirs = [d for d in current_dir.iterdir() if d.is_dir()]

    print(f"Found {len(depth1_dirs)} directories to process")

    all_detailed_results = []
    for dir_path in depth1_dirs:
        result = process_single_directory(dir_path)
        if result is not None:
            all_detailed_results.extend(result)

    # Create DataFrame and save to CSV
    if all_detailed_results:
        df = pd.DataFrame(all_detailed_results)
        output_filename = "pose_analysis.csv"
        df.to_csv(output_filename, index=False)
        print(f"\nSaved detailed pose analysis to {output_filename}")
        print(f"Total poses analyzed: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # Display summary statistics
        print(f"\nSummary:")
        print(f"- Total systems processed: {len(set(df['PDB_ID']))}")
        print(f"- Total poses: {len(df)}")
        print(f"- Average RMSD: {df['RMSD'].mean():.4f}")
        print(f"- Min RMSD: {df['RMSD'].min():.4f}")
        print(f"- Max RMSD: {df['RMSD'].max():.4f}")
        print(f"- Average Vina Score: {df['Vina_Score'].mean():.3f}")
        print(f"- Average Vina Rank: {df['Vina_Rank'].mean():.2f}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
