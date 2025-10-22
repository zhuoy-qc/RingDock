import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, cpu_count
import time
import math
from rdkit import Chem
from rdkit import RDLogger
from plip.structure.preparation import PDBComplex

# Suppress RDKit warnings globally (including the 2D/3D warning)
RDLogger.DisableLog('rdApp.*')

def calculate_angle(v1, v2):
    """Calculate angle between two vectors in degrees.
    If angle > 90, return 180 - angle to reflect acute orientation."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    if angle > 90:
        angle = 180 - angle
    return angle

def calculate_rz(distance, offset):
    """Calculate RZ value (projection of distance on Z-axis)"""
    return np.sqrt(distance**2 - offset**2)

def compute_dihedral_angle(p1, p2, p3, n2):
    """Compute dihedral angle between plane defined by p1-p2-p3 and ring normal n2."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    n2 = np.array(n2, dtype=float)

    v1 = p2 - p1
    v2 = p3 - p1
    n1 = np.cross(v1, v2)
    norm_n1 = np.linalg.norm(n1)
    if norm_n1 < 1e-8:
        raise ValueError("Charged atoms collinear")
    n1 = n1 / norm_n1

    norm_n2 = np.linalg.norm(n2)
    if norm_n2 < 1e-8:
        raise ValueError("Ring normal zero")
    n2 = n2 / norm_n2

    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(np.abs(cos_angle))
    return np.degrees(angle_rad)

def create_complex_pdb(subdir_path):
    subdir = Path(subdir_path)
    subdir_name = subdir.name

    try:
        ligand_files = list(subdir.glob('*_ligand.sdf'))
        if not ligand_files:
            return False, f"No *_ligand.sdf file found", subdir_name
        ligand_file = ligand_files[0]

        protein_files = list(subdir.glob('*_protein.pdb'))
        if not protein_files:
            return False, f"No *_only_protein.pdb file found", subdir_name
        protein_file = protein_files[0]

        output_file = subdir / 'complex.pdb'

        with open(protein_file, 'r') as prot:
            protein_content = prot.readlines()

        supplier = Chem.SDMolSupplier(str(ligand_file), sanitize=False)
        mol = None
        for m in supplier:
            if m is not None:
                mol = m
                break

        if mol is None:
            return False, f"Could not read valid molecule from {ligand_file.name}", subdir_name

        ligand_pdb_block = Chem.MolToPDBBlock(mol)

        with open(output_file, 'w') as out:
            for line in protein_content:
                if line.startswith(('ATOM', 'HETATM')):
                    out.write(line)

            for line in ligand_pdb_block.split('\n'):
                if line.startswith('ATOM'):
                    hetatm_line = 'HETATM' + line[6:]
                    out.write(hetatm_line + '\n')
                elif line.startswith('HETATM'):
                    out.write(line + '\n')
                elif line.startswith('CONECT'):
                    out.write(line + '\n')

            out.write("END\n")

        return True, f"Successfully created {output_file.name}", subdir_name

    except Exception as e:
        return False, f"Error: {str(e)}", subdir_name

def analyze_pication_interactions(pdb_file_path):
    from io import StringIO
    import sys
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        pdb_file = Path(pdb_file_path)
        dir_name = pdb_file.parent.name

        my_mol = PDBComplex()
        my_mol.load_pdb(str(pdb_file))
        my_mol.analyze()

        results = []

        if not hasattr(my_mol, 'interaction_sets') or not my_mol.interaction_sets:
            sys.stderr = old_stderr
            return results, "No interaction sets found", dir_name

        for bs_id, interactions in my_mol.interaction_sets.items():
            if hasattr(interactions, 'pication_laro'):
                for pication in interactions.pication_laro:
                    try:
                        ring_normal = np.array(pication.ring.normal, dtype=np.float64)
                        charge_vector = np.array(pication.charge.center, dtype=np.float64) - np.array(pication.ring.center, dtype=np.float64)
                        angle = calculate_angle(ring_normal, charge_vector)
                        rz = calculate_rz(pication.distance, pication.offset)

                        protein_res = pication.restype.strip().upper()
                        is_arg = (protein_res == 'ARG')
                        dihedral = float('nan')

                        if is_arg:
                            charge_atoms = pication.charge.atoms
                            if len(charge_atoms) >= 3:
                                try:
                                    coords = [atom.coords for atom in charge_atoms[:3]]
                                    dihedral = compute_dihedral_angle(coords[0], coords[1], coords[2], ring_normal)
                                except Exception:
                                    dihedral = float('nan')

                        interaction_data = {
                            'Directory': dir_name,
                            'PDB_File': pdb_file.name,
                            'Binding_Site': bs_id,
                            'Ligand': f"{pication.restype_l}-{pication.resnr_l}-{pication.reschain_l}",
                            'Protein': f"{pication.restype}-{pication.resnr}-{pication.reschain}",
                            'Protein_Residue_Type': protein_res,
                            'Is_ARG': is_arg,
                            'Distance_Å': float(pication.distance),
                            'Offset_Å': float(pication.offset),
                            'RZ_Å': rz,
                            'Angle_°': angle,
                            'Dihedral_Angle_°': dihedral,
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
                        pass  # Silent failure for individual interactions

        sys.stderr = old_stderr
        return results, f"Found {len(results)} π-cation interactions", dir_name

    except Exception as e:
        sys.stderr = old_stderr
        return [], f"Analysis error: {str(e)}", dir_name

def process_single_directory(subdir_path):
    subdir = Path(subdir_path)
    results = {
        'directory': subdir.name,
        'complex_created': False,
        'analysis_success': False,
        'interactions': [],
        'messages': []
    }

    success, message, _ = create_complex_pdb(subdir_path)
    results['complex_created'] = success
    results['messages'].append(f"Complex creation: {message}")

    if not success:
        return results

    complex_file = subdir / 'complex.pdb'
    if complex_file.exists():
        interactions, message, _ = analyze_pication_interactions(str(complex_file))
        results['analysis_success'] = len(interactions) > 0
        results['interactions'] = interactions
        results['messages'].append(f"Analysis: {message}")

    return results

def generate_csv_report(all_results, output_file="new_reference_experimental_pication_interactions_report.csv"):
    all_interactions = []
    for result in all_results:
        all_interactions.extend(result['interactions'])

    if not all_interactions:
        print("No π-cation interactions found for CSV report")
        return None

    df = pd.DataFrame(all_interactions)

    column_order = [
        'Directory', 'PDB_File', 'Binding_Site', 'Ligand', 'Protein',
        'Protein_Residue_Type', 'Is_ARG',
        'Distance_Å', 'Offset_Å', 'RZ_Å', 'Angle_°', 'Dihedral_Angle_°',
        'Ring_Center_X', 'Ring_Center_Y', 'Ring_Center_Z',
        'Charged_Center_X', 'Charged_Center_Y', 'Charged_Center_Z',
        'Ring_Normal_X', 'Ring_Normal_Y', 'Ring_Normal_Z',
        'Ring_Type', 'Atom_Indices', 'Interaction_Type'
    ]

    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    df.to_csv(output_file, index=False, float_format='%.3f')
    return output_file

def save_directories_with_interactions(all_results, output_file="directories_with_interactions.txt"):
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

def main():
    start_time = time.time()
    current_dir = Path.cwd()
    subdirs = [str(d) for d in current_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print("No subdirectories found in current directory")
        return

    print(f"Found {len(subdirs)} subdirectories for processing")

    max_cores = 96
    available_cores = cpu_count()
    num_processes = min(len(subdirs), available_cores, max_cores)
    print(f"Using {num_processes} parallel processes (available: {available_cores}, max: {max_cores})")

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_directory, subdirs)

    csv_file = generate_csv_report(results)
    interaction_file, directories_with_interactions = save_directories_with_interactions(results)

    total_dirs = len(results)
    successful_creations = sum(1 for r in results if r['complex_created'])
    successful_analyses = sum(1 for r in results if r['analysis_success'])
    total_interactions = sum(len(r['interactions']) for r in results)

    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total directories processed: {total_dirs}")
    print(f"Successful complex creations: {successful_creations}")
    print(f"Directories with π-cation interactions: {successful_analyses}")
    print(f"Total π-cation interactions found: {total_interactions}")
    print(f"Directories with interactions saved to: {interaction_file}")
    print(f"Detailed report generated: {csv_file}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per directory: {elapsed_time/total_dirs:.2f} seconds")

    # Only print directories WITH interactions (skip 0-interaction entries)
    if directories_with_interactions:
        print(f"\n{'='*60}")
        print("DIRECTORIES WITH π-CATION INTERACTIONS")
        print(f"{'='*60}")
        for directory in directories_with_interactions:
            count = next((len(r['interactions']) for r in results if r['directory'] == directory), 0)
            print(f"✓ {directory}: {count} interactions")

if __name__ == "__main__":
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

    multiprocessing.freeze_support()
    main()
