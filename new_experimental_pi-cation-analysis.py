#!/usr/bin/env python3
"""
Creates complex.pdb files and analyzes π-cation interactions across multiple directories
Removes temporary PLIP files and purifies protein files using BioPython
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, cpu_count
import time
import re
from rdkit import Chem
from plip.structure.preparation import PDBComplex
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa

def calculate_angle(v1, v2):
    """Calculate angle between two vectors in degrees.
    If angle > 90, return 180 - angle to reflect acute orientation."""
    # Normalize the vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    # Compute the angle in degrees
    angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    # Adjust if angle is obtuse
    if angle > 90:
        angle = 180 - angle

    return angle

def calculate_rz(distance, offset):
    """Calculate RZ value (projection of distance on Z-axis)"""
    return np.sqrt(distance**2 - offset**2)

class ProteinSelect(Select):
    """Custom BioPython selector to keep only standard amino acid residues"""
    def accept_model(self, model):
        """Accept all models"""
        return True

    def accept_chain(self, chain):
        """Accept all chains"""
        return True

    def accept_residue(self, residue):
        """Only accept standard amino acid residues"""
        return is_aa(residue, standard=True)

def clean_pdb(input_file, output_file='pure_protein.pdb'):
    """
    Read PDB file, remove all non-protein residues, and save pure protein structure.

    Parameters:
        input_file (str): Input PDB file path
        output_file (str): Output pure protein PDB file path (default: 'pure_protein.pdb')
    """
    try:
        parser = PDBParser(QUIET=True)  # Suppress warnings
        structure = parser.get_structure('protein', input_file)

        io = PDBIO()
        io.set_structure(structure)
        io.save(output_file, ProteinSelect())
        return True, f"Purified protein structure saved to {output_file}"
    except Exception as e:
        return False, f"Error purifying protein: {str(e)}"

def find_protein_file(subdir_path):
    """
    Search for protein files in the directory following this priority:
    1. *_only_protein.pdb
    2. *_protein.pdb
    """
    subdir = Path(subdir_path)

    # First, look for *_only_protein.pdb files
    only_protein_files = list(subdir.glob('*_only_protein.pdb'))
    if only_protein_files:
        return str(only_protein_files[0])

    # If not found, look for *_protein.pdb files
    protein_files = list(subdir.glob('*_protein.pdb'))
    if protein_files:
        return str(protein_files[0])

    # If no protein files found
    return None

def create_complex_pdb(subdir_path):
    """
    Create complex.pdb from *_ligand.sdf and *_only_protein.pdb or *_protein.pdb in a directory
    """
    subdir = Path(subdir_path)
    subdir_name = subdir.name

    try:
        # Find ligand SDF file
        ligand_files = list(subdir.glob('*_ligand.sdf'))
        if not ligand_files:
            return False, f"No *_ligand.sdf file found", subdir_name
        ligand_file = ligand_files[0]

        # Find protein PDB file (first *_only_protein.pdb, then *_protein.pdb)
        protein_file_path = find_protein_file(subdir_path)
        if not protein_file_path:
            return False, f"No *_only_protein.pdb or *_protein.pdb file found", subdir_name
        protein_file = Path(protein_file_path)

        # Output file path
        output_file = subdir / 'complex.pdb'

        # Read protein content
        with open(protein_file, 'r') as prot:
            protein_content = prot.readlines()

        # Read ligand and convert to PDB
        supplier = Chem.SDMolSupplier(str(ligand_file), sanitize=False)
        mol = None
        for m in supplier:
            if m is not None:
                mol = m
                break

        if mol is None:
            return False, f"Could not read valid molecule from {ligand_file.name}", subdir_name

        ligand_pdb_block = Chem.MolToPDBBlock(mol)

        # Write complex file
        with open(output_file, 'w') as out:
            # Write protein content
            for line in protein_content:
                if line.startswith(('ATOM', 'HETATM')):
                    out.write(line)

            # Write ligand content with HETATM labels
            for line in ligand_pdb_block.split('\n'):
                if line.startswith('ATOM'):
                    # Convert ATOM to HETATM
                    hetatm_line = 'HETATM' + line[6:]
                    out.write(hetatm_line)
                elif line.startswith('HETATM'):
                    out.write(line + '\n')
                elif line.startswith('CONECT'):
                    out.write(line + '\n')

            out.write("END\n")

        return True, f"Successfully created {output_file.name}", subdir_name

    except Exception as e:
        return False, f"Error: {str(e)}", subdir_name

def analyze_pication_interactions(pdb_file_path):
    """
    Analyze π-cation interactions in a complex.pdb file with detailed geometric parameters
    """
    pdb_file = Path(pdb_file_path)
    dir_name = pdb_file.parent.name

    try:
        my_mol = PDBComplex()
        my_mol.load_pdb(str(pdb_file))
        my_mol.analyze()

        results = []

        if not hasattr(my_mol, 'interaction_sets') or not my_mol.interaction_sets:
            return results, "No interaction sets found", dir_name

        for bs_id, interactions in my_mol.interaction_sets.items():
            if hasattr(interactions, 'pication_laro'):
                for pication in interactions.pication_laro:
                    try:
                        ring_normal = np.array(pication.ring.normal, dtype=np.float64)
                        charge_vector = np.array(pication.charge.center, dtype=np.float64) - np.array(pication.ring.center, dtype=np.float64)
                        angle = calculate_angle(ring_normal, charge_vector)
                        rz = calculate_rz(pication.distance, pication.offset)

                                                interaction_data = {
                            'Directory': dir_name,
                            'PDB_File': pdb_file.name,
                            'Binding_Site': bs_id,
                            'Ligand': f"{pication.restype_l}-{pication.resnr_l}-{pication.reschain_l}",
                            'Protein': f"{pication.restype}-{pication.resnr}-{pication.reschain}",
                            'Distance_Å': float(pication.distance),
                            'Offset_Å': float(pication.offset),
                            'RZ_Å': rz,
                            'Angle_°': angle,
                            'Ring_Center_X': float(pication.ring.center[0]),
                            'Ring_Center_Y': float(pication.ring.center[1]),
                            'Ring_Center_Z': float(pication.ring.center[2]),
                            'Charged_Center_X': float(pication.charge.center[0]),
                            'Charged_Center_Y': float(pication.charge.center[1]),
                            'Charged_Center_Z': float(pication.charge.center[2]),
                            'Ring_Normal_X': float(ring_normal[0]),
                            'Ring_Normal_Y': float(ring_normal[1]),
                            'Ring_Normal_Z': float(ring_normal[2]),
                            'Ring_Type': pication.ring.type,
                            'Atom_Indices': str(pication.ring.atoms_orig_idx),
                            'Interaction_Type': 'π-Cation'
                        }

                        results.append(interaction_data)
                    except Exception as e:
                        print(f"Error processing π-cation interaction: {e}")

        return results, f"Found {len(results)} π-cation interactions", dir_name

    except Exception as e:
        return [], f"Analysis error: {str(e)}", dir_name

def process_single_directory(subdir_path):
    """
    Complete processing pipeline for a single directory
    """
    subdir = Path(subdir_path)
    results = {
        'directory': subdir.name,
        'complex_created': False,
        'analysis_success': False,
        'interactions': [],
        'messages': [],
        'protein_purified': False
    }

    # Step 1: Find and purify protein file if exists
    protein_file_path = find_protein_file(subdir_path)
    if protein_file_path:
        protein_file = Path(protein_file_path)
        pure_protein_file = subdir / 'pure_protein.pdb'
        success, message = clean_pdb(str(protein_file), str(pure_protein_file))
        results['protein_purified'] = success
        results['messages'].append(f"Protein purification: {message}")

        # If purification was successful, use the pure protein file for complex creation
        if success:
            # Update the protein file to use the purified version
            original_protein_file = protein_file_path
            # Rename the original file temporarily
            original_backup = subdir / f"{protein_file.stem}_original{protein_file.suffix}"
            if original_backup != protein_file:
                protein_file.rename(original_backup)

            # Rename the pure protein file to the expected name
            pure_protein_file.rename(protein_file)

    # Step 2: Create complex.pdb
    success, message, _ = create_complex_pdb(subdir_path)
    results['complex_created'] = success
    results['messages'].append(f"Complex creation: {message}")

    if not success:
        return results

    # Step 3: Analyze π-cation interactions
    complex_file = subdir / 'complex.pdb'
    if complex_file.exists():
        interactions, message, _ = analyze_pication_interactions(str(complex_file))
        results['analysis_success'] = len(interactions) > 0
        results['interactions'] = interactions
        results['messages'].append(f"Analysis: {message}")

    return results

def generate_csv_report(all_results, output_file="pication_interactions_report.csv"):
    """
    Generate a detailed CSV report with all π-cation interaction data
    """
    all_interactions = []
    for result in all_results:
        all_interactions.extend(result['interactions'])

    if not all_interactions:
        print("No π-cation interactions found for CSV report")
        return None

        # Create DataFrame
    df = pd.DataFrame(all_interactions)

    # Reorder columns for better readability
    column_order = [
        'Directory', 'PDB_File', 'Binding_Site', 'Ligand', 'Protein',
        'Distance_Å', 'Offset_Å', 'RZ_Å', 'Angle_°',
        'Ring_Center_X', 'Ring_Center_Y', 'Ring_Center_Z',
        'Charged_Center_X', 'Charged_Center_Y', 'Charged_Center_Z',
        'Ring_Normal_X', 'Ring_Normal_Y', 'Ring_Normal_Z',
        'Ring_Type', 'Atom_Indices', 'Interaction_Type'
    ]

    # Keep only columns that exist in the DataFrame
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    # Save to CSV
    df.to_csv(output_file, index=False, float_format='%.3f')
    return output_file

def save_directories_with_interactions(all_results, output_file="directories_with_interactions.txt"):
    """
    Save list of directories that have at least one π-cation interaction
    """
    directories_with_interactions = []
    for result in all_results:
        if result['interactions']:
            directories_with_interactions.append(result['directory'])

    if directories_with_interactions:
        with open(output_file, 'w') as f:
            for directory in directories_with_interactions:
                f.write(f"{directory}\n")
        return output_file, directories_with_interactions
    else:
        return None, []

def cleanup_temp_files():
    """
    Remove temporary PLIP files (plipfixed.complex_*.pdb) from current directory
    """
    current_dir = Path.cwd()
    temp_files = list(current_dir.glob('plipfixed.complex_*.pdb'))

    # Also look for files in subdirectories
    for subdir in current_dir.iterdir():
        if subdir.is_dir():
            temp_files.extend(subdir.glob('plipfixed.complex_*.pdb'))

    deleted_count = 0
    for temp_file in temp_files:
        try:
            temp_file.unlink()
            deleted_count += 1
            print(f"Removed temporary file: {temp_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")

    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} temporary PLIP files")
    else:
        print("No temporary PLIP files found to clean up")

def main():
    """Main function to run parallel processing pipeline"""
    start_time = time.time()

    # Get current directory and subdirectories
    current_dir = Path.cwd()
    subdirs = [str(d) for d in current_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print("No subdirectories found in current directory")
        return

    print(f"Found {len(subdirs)} subdirectories for processing")

    # Determine number of processes (max 96 cores)
    max_cores = 96
    available_cores = cpu_count()
    num_processes = min(len(subdirs), available_cores, max_cores)
    print(f"Using {num_processes} parallel processes (available: {available_cores}, max: {max_cores})")

    # Process directories in parallel
    print("Starting parallel processing...")
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_directory, subdirs)

    # Generate CSV report
    csv_file = generate_csv_report(results)

    # Save directories with interactions
    interaction_file, directories_with_interactions = save_directories_with_interactions(results)

    # Clean up temporary PLIP files
    print("\nCleaning up temporary files...")
    cleanup_temp_files()

    # Print summary to console
    total_dirs = len(results)
    successful_creations = sum(1 for r in results if r['complex_created'])
    successful_analyses = sum(1 for r in results if r['analysis_success'])
        total_interactions = sum(len(r['interactions']) for r in results)
    purified_proteins = sum(1 for r in results if r['protein_purified'])

    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total directories processed: {total_dirs}")
    print(f"Successful complex creations: {successful_creations}")
    print(f"Directories with π-cation interactions: {successful_analyses}")
    print(f"Total π-cation interactions found: {total_interactions}")
    print(f"Protein files purified: {purified_proteins}")
    print(f"Directories with interactions saved to: {interaction_file}")
    print(f"Detailed report generated: {csv_file}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per directory: {elapsed_time/total_dirs:.2f} seconds")

    # Print individual directory status
    print(f"\n{'='*60}")
    print("DETAILED STATUS BY DIRECTORY")
    print(f"{'='*60}")
    for result in results:
        status = "✓" if result['analysis_success'] else "✗" if result['complex_created'] else "⚠"
        purification_status = "P" if result['protein_purified'] else "N"
        print(f"{status} {result['directory']}: {len(result['interactions'])} interactions [Purification: {purification_status}]")

    # Print directories with interactions
    if directories_with_interactions:
        print(f"\n{'='*60}")
        print("DIRECTORIES WITH π-CATION INTERACTIONS")
        print(f"{'='*60}")
        for directory in directories_with_interactions:
            print(f"✓ {directory}")

if __name__ == "__main__":
    # Check for required packages
    try:
        from rdkit import Chem
    except ImportError:
        print("Error: RDKit not installed. Install with: conda install -c conda-forge rdkit")
        sys.exit(1)

    try:
        from plip.structure.preparation import PDBComplex
    except ImportError:
        print("Error: PLIP not installed. Install with: pip install plip")
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas not installed. Install with: pip install pandas")
        sys.exit(1)

    try:
        from Bio.PDB import PDBParser, PDBIO, Select
        from Bio.PDB.Polypeptide import is_aa
    except ImportError:
        print("Error: BioPython not installed. Install with: pip install biopython")
        sys.exit(1)

    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()

    # Run the pipeline
    main()
