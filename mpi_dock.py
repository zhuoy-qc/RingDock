import os
import glob
import re
import subprocess
import logging
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
import multiprocessing
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_and_protonate_pdb_files(base_dir):
    """
    Find *_*_protein.pdb files in subdirectories (depth=1) and protonate them using obabel.
    Skips files containing 'only' in their filename.
    """
    protonated_files = []
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        pdb_pattern = os.path.join(subdir_path, "*_*_protein.pdb")
        pdb_files = glob.glob(pdb_pattern)

        for pdb_file in pdb_files:
            if "only" in os.path.basename(pdb_file).lower():
                logger.info(f"Skipping file containing 'only': {pdb_file}")
                continue

            try:
                base_name = os.path.basename(pdb_file).replace('.pdb', '')
                protonated_file = os.path.join(subdir_path, f"{base_name}_protonated.pdb")
                cmd = ['obabel', pdb_file, '-O', protonated_file, '-h', '--pdb']  # Protonate

                logger.info(f"Protonating {pdb_file} -> {protonated_file}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    if os.path.exists(protonated_file) and os.path.getsize(protonated_file) > 0:
                        protonated_files.append(protonated_file)
                        logger.info(f"Successfully protonated: {protonated_file}")
                    else:
                        logger.warning(f"Protonation failed for {pdb_file}: output file not created or empty")
                else:
                    logger.error(f"Obabel error for {pdb_file}: {result.stderr}")

            except Exception as e:
                logger.error(f"Error processing {pdb_file}: {e}")
                continue

    return protonated_files

def classify_aromatic_rings(sdf_file, reference_ring_dir="/scratch_sh/zyin/ring_sdf_files"):
    """
    Analyze the molecule in the SDF file, identify and classify aromatic ring systems,
    and match them against reference rings.
    """
    suppl = Chem.SDMolSupplier(sdf_file)
    molecule = next(suppl, None)
    if molecule is None:
        raise ValueError(f"Could not read molecule from {sdf_file}")

    molecule = Chem.AddHs(molecule)
    Chem.SanitizeMol(molecule)
    aromatic_rings_info = []
    ring_info = molecule.GetRingInfo()
    atoms = molecule.GetAtoms()

    # Identify all aromatic rings
    aromatic_rings = []
    for ring in ring_info.AtomRings():
        if all(atoms[idx].GetIsAromatic() for idx in ring):
            aromatic_rings.append(frozenset(ring))

    # Classify single aromatic rings
    for ring_set in aromatic_rings:
        ring_list = list(ring_set)
        elements = [atoms[idx].GetSymbol() for idx in ring_list]
        ring_size = len(ring_list)

        if ring_size == 6:
            n_count = elements.count('N')
            if n_count == 0:
                aromatic_rings_info.append(('Benzene', ring_list))
            elif n_count == 1:
                aromatic_rings_info.append(('Pyridine', ring_list))
            elif n_count == 2:
                positions = []
                for i, idx in enumerate(ring_list):
                    if atoms[idx].GetSymbol() == 'N':
                        positions.append(i)
                if abs(positions[0] - positions[1]) == 1 or abs(positions[0] - positions[1]) == 5:
                    aromatic_rings_info.append(('Pyridazine-like', ring_list))
                elif abs(positions[0] - positions[1]) == 2 or abs(positions[0] - positions[1]) == 4:
                    aromatic_rings_info.append(('Pyrimidine-like', ring_list))
                else:
                    aromatic_rings_info.append(('Pyrazine-like', ring_list))
            elif n_count >= 3:
                aromatic_rings_info.append(('Triazine/Tetrazine', ring_list))
        elif ring_size == 5:
            heteroatoms = [e for e in elements if e != 'C']
            if not heteroatoms:
                aromatic_rings_info.append(('Cyclopentadiene-like', ring_list))
            elif 'N' in heteroatoms and 'O' not in heteroatoms and 'S' not in heteroatoms:
                aromatic_rings_info.append(('Pyrrole-like', ring_list))
            elif 'O' in heteroatoms:
                aromatic_rings_info.append(('Furan-like', ring_list))
            elif 'S' in heteroatoms:
                aromatic_rings_info.append(('Thiophene-like', ring_list))
            else:
                aromatic_rings_info.append(('Other 5-membered heteroaromatic', ring_list))
        else:
            aromatic_rings_info.append((f'Other {ring_size}-membered aromatic ring', ring_list))

    # Identify fused bicyclic systems (simplified for brevity)
    fused_systems = []
    for i, ring_a in enumerate(aromatic_rings):
        for ring_b in aromatic_rings[i+1:]:
            if ring_a & ring_b:
                found = False
                for sys in fused_systems:
                    if ring_a in sys or ring_b in sys:
                        sys.update([ring_a, ring_b])
                        found = True
                        break
                if not found:
                    fused_systems.append({ring_a, ring_b})

    bicyclic_fused_info = []
    for sys in fused_systems:
        if len(sys) == 2:
            ring_a, ring_b = list(sys)
            size_a, size_b = len(ring_a), len(ring_b)
            elements_a = [atoms[idx].GetSymbol() for idx in ring_a]
            elements_b = [atoms[idx].GetSymbol() for idx in ring_b]
            n_count = elements_a.count('N') + elements_b.count('N')

            if size_a == 6 and size_b == 6:
                if n_count == 0:
                    bicyclic_fused_info.append(('Naphthalene-like', list(ring_a), list(ring_b)))
                elif n_count == 1:
                    bicyclic_fused_info.append(('Quinoline/Isoquinoline-like', list(ring_a), list(ring_b)))
                elif n_count == 2:
                    bicyclic_fused_info.append(('Cinnoline/Phthalazine-like', list(ring_a), list(ring_b)))
                else:
                    bicyclic_fused_info.append((f'Fused 6+6 ring with {n_count} N atoms', list(ring_a), list(ring_b)))
            elif {size_a, size_b} == {5, 6}:
                five_ring = ring_a if size_a == 5 else ring_b
                six_ring = ring_b if size_b == 6 else ring_a
                five_elements = [atoms[idx].GetSymbol() for idx in five_ring]
                heteroatoms = [e for e in five_elements if e != 'C']
                if 'N' in heteroatoms:
                    bicyclic_fused_info.append(('Indole-like', list(five_ring), list(six_ring)))
                elif 'O' in heteroatoms:
                    bicyclic_fused_info.append(('Benzofuran-like', list(five_ring), list(six_ring)))
                elif 'S' in heteroatoms:
                    bicyclic_fused_info.append(('Benzothiophene-like', list(five_ring), list(six_ring)))
                else:
                    bicyclic_fused_info.append(('Other 5+6 fused system', list(five_ring), list(six_ring)))
            else:
                bicyclic_fused_info.append((f'Other fused system ({size_a}+{size_b})', list(ring_a), list(ring_b)))

    # Match with reference rings (if reference directory exists)
    ring_matches = []
    if reference_ring_dir and os.path.exists(reference_ring_dir):
        reference_mapping = {
            'benzene.sdf': 'Benzene',
            'pyridine.sdf': 'Pyridine',
            'pyrimidine.sdf': 'Pyrimidine',
            'pyrrole.sdf': 'Pyrrole',
            'thiophene.sdf': 'Thiophene'
        }
        for ring_type, atom_indices in aromatic_rings_info:
            matched_ref = None
            matched_path = None
            for ref_file, ref_name in reference_mapping.items():
                if ring_type.lower().startswith(ref_name.lower()):
                    ref_path = os.path.join(reference_ring_dir, ref_file)
                    if os.path.exists(ref_path):
                        matched_ref = ref_name
                        matched_path = ref_path
                        break
            if matched_ref and matched_path:
                ring_matches.append((ring_type, atom_indices, matched_ref, matched_path))

    return aromatic_rings_info, bicyclic_fused_info, ring_matches

def run_smina_docking(protein_file, ligand_file, autobox_ligand, output_dir, exhaustiveness=80):
    """
    Run Smina docking for a given protein and ligand.
    """
    try:
        protein_name = os.path.basename(protein_file).replace('_protonated.pdb', '')
        ligand_name = os.path.basename(ligand_file).replace('.sdf', '')
        output_file = os.path.join(output_dir, f"{protein_name}_{ligand_name}_docked.sdf")

        cmd = [
            'smina', '-r', protein_file, '-l', ligand_file,
            '--autobox_ligand', autobox_ligand, '-o', output_file,
            '--exhaustiveness', str(exhaustiveness), '--seed', '1',
            '--num_modes', '100', '--scoring', 'vinardo', '--energy_range', '5', '--min_rmsd_filter', '1'
        ]

        logger.info(f"Running Smina docking: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"Smina docking completed: {output_file}")
                return output_file
            else:
                logger.error(f"Smina output file not created or empty for {ligand_file}")
                return None
        else:
            logger.error(f"Smina error for {ligand_file}: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Error running Smina for {ligand_file}: {e}")
        return None

def find_ligand_sdf_files(directory):
    """Find all ligand SDF files in the given directory."""
    ligand_pattern = os.path.join(directory, "*_ligand.sdf")
    return glob.glob(ligand_pattern)

def setup_environment():
    """Check if necessary tools (OpenBabel, Smina) are available."""
    try:
        subprocess.run(['obabel', '-V'], capture_output=True, check=True)
        logger.info("OpenBabel is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("OpenBabel is not installed or not in PATH")
        return False

    try:
        subprocess.run(['smina', '--help'], capture_output=True, check=True)
        logger.info("Smina is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Smina is not installed or not in PATH")
        return False

    return True

def process_single_protein(protonated_file, reference_ring_dir):
    """
    Process a single protonated protein file: find ligands, classify rings, and run docking.
    This function is designed to be called in parallel.
    """
    protein_dir = os.path.dirname(protonated_file)
    ligand_files = find_ligand_sdf_files(protein_dir)
    if not ligand_files:
        logger.warning(f"No ligand files found in {protein_dir}")
        return

    autobox_ligand = ligand_files[0]  # Use the first ligand for autobox reference
    logger.info(f"Using {autobox_ligand} for autobox reference in {protein_dir}")

    # Classify aromatic rings for each ligand and collect all matches
    all_ring_matches = []
    for ligand_file in ligand_files:
        try:
            aromatic_rings, bicyclic_fused, ring_matches = classify_aromatic_rings(ligand_file, reference_ring_dir)
            if ring_matches:
                all_ring_matches.extend(ring_matches)
                logger.info(f"Found {len(ring_matches)} ring matches in {ligand_file}")
        except Exception as e:
            logger.error(f"Error in ring matching for {ligand_file}: {e}")
            continue

    if not all_ring_matches:
        logger.warning(f"No ring matches found in {protein_dir}")
        return

    # Run docking for each ring match
    docking_results = []
    for ring_match in all_ring_matches:
        ring_type, atom_indices, ref_name, ref_path = ring_match
        docking_result = run_smina_docking(
            protein_file=protonated_file,
            ligand_file=ref_path,
            autobox_ligand=autobox_ligand,
            output_dir=protein_dir
        )
        if docking_result:
            docking_results.append((ref_name, docking_result))
            logger.info(f"Successfully completed docking for {ref_name} ring in {protein_dir}")
        else:
            logger.error(f"Docking failed for {ref_name} ring in {protein_dir}")

    return docking_results

def main_workflow():
    """
    Main workflow: parallelize the processing of each protonated protein file across available CPU cores.
    """
    logger.info("Starting automated molecular docking workflow...")

    base_dir = os.getcwd()
    reference_ring_dir = "/scratch_sh/zyin/ring_sdf_files"

    if not os.path.exists(reference_ring_dir):
        logger.warning(f"Reference ring directory not found: {reference_ring_dir}. Ring matching disabled.")
        reference_ring_dir = None

    if not setup_environment():
        logger.error("Environment setup failed. Please install required tools.")
        return

    # Step 1: Find and protonate PDB files
    logger.info("Step 1: Finding and protonating PDB files")
    protonated_files = find_and_protonate_pdb_files(base_dir)
    if not protonated_files:
        logger.error("No protonated PDB files found. Exiting.")
        return
    logger.info(f"Found {len(protonated_files)} protonated PDB files")

    # Step 2: Parallel processing of each protonated file
    logger.info("Step 2: Parallel processing of protonated files")
    num_cores = cpu_count()
    # Use all available cores for parallel processing
    with Pool(processes=num_cores) as pool:
        # Prepare arguments for each process: (protonated_file, reference_ring_dir)
        tasks = [(pf, reference_ring_dir) for pf in protonated_files]
        results = pool.starmap(process_single_protein, tasks)

    logger.info("Automated molecular docking workflow completed!")

if __name__ == "__main__":
    # Set the start method for multiprocessing to 'spawn' for compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main_workflow()
