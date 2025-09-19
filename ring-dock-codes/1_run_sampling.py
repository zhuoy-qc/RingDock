import os
import glob
import subprocess
import logging
from multiprocessing import Pool, cpu_count
import multiprocessing
from tqdm import tqdm
import time
import psutil
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURABLE PARAMETERS ===
# Easy to change these two numbers:
CONCURRENT_PROTEINS = 12      # Number of proteins to process simultaneously
CPU_CORES_PER_SMINA = 8     # Number of CPU cores each Smina job will use
TIMEOUT_SECONDS = 120        # Timeout in seconds (2 minutes)
# ===============================

# Global variable for timeout log file
TIMEOUT_LOG_FILE = "smina_timeouts.log"

def protonate_single_file(args):
    """
    Protonate a single PDB file using obabel with optimized performance.
    """
    pdb_file, subdir_path = args
    
    if "only" in os.path.basename(pdb_file).lower():
        logger.info(f"Skipping file containing 'only': {pdb_file}")
        return None

    try:
        base_name = os.path.basename(pdb_file).replace('.pdb', '')
        protonated_file = os.path.join(subdir_path, f"{base_name}_protonated.pdb")
        
        cmd = [
            'obabel', 
            pdb_file, 
            '-O', protonated_file, 
            '-h',           
            '--pdb'         
        ]

        logger.debug(f"Protonating {pdb_file}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and os.path.exists(protonated_file) and os.path.getsize(protonated_file) > 0:
            logger.info(f"Successfully protonated: {protonated_file}")
            return protonated_file
        else:
            error_msg = result.stderr if result.returncode != 0 else "Output file not created or empty"
            logger.warning(f"Protonation failed for {pdb_file}: {error_msg}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout for {pdb_file}")
        return None
    except Exception as e:
        logger.error(f"Error processing {pdb_file}: {e}")
        return None

def find_and_protonate_pdb_files(base_dir):
    """
    Find and protonate PDB files with maximum parallelization.
    """
    pdb_files_to_process = []
    
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        pdb_pattern = os.path.join(subdir_path, "*_*_protein.pdb")
        pdb_files = glob.glob(pdb_pattern)
        pdb_files_to_process.extend([(pdb_file, subdir_path) for pdb_file in pdb_files])

    if not pdb_files_to_process:
        logger.info("No PDB files found to process")
        return []

    logger.info(f"Found {len(pdb_files_to_process)} PDB files to protonate")

    # Use 75% of cores for protonation to leave room for other processes
    num_cores = max(1, int(cpu_count() * 0.75))
    logger.info(f"Using {num_cores} cores for parallel protonation (75% of {cpu_count()} total)")

    with Pool(processes=num_cores, maxtasksperchild=5) as pool:
        protonated_files = list(tqdm(
            pool.imap(protonate_single_file, pdb_files_to_process, chunksize=1),
            total=len(pdb_files_to_process),
            desc="Protonating PDB files",
            unit="file"
        ))

    successful_files = [f for f in protonated_files if f is not None]
    logger.info(f"Successfully protonated {len(successful_files)} out of {len(pdb_files_to_process)} files")
    return successful_files

def classify_aromatic_rings(sdf_file, reference_ring_dir="/scratch_sh/zyin/ring_sdf_files"):
    """
    Analyze the molecule in the SDF file, identify and classify aromatic ring systems,
    and match them against reference rings.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        suppl = Chem.SDMolSupplier(sdf_file)
        molecule = next(suppl, None)
        if molecule is None:
            raise ValueError(f"Could not read molecule from {sdf_file}")

        molecule = Chem.AddHs(molecule)
        Chem.SanitizeMol(molecule)
        aromatic_rings_info = []
        ring_info = molecule.GetRingInfo()
        atoms = molecule.GetAtoms()

        aromatic_rings = []
        for ring in ring_info.AtomRings():
            if all(atoms[idx].GetIsAromatic() for idx in ring):
                aromatic_rings.append(frozenset(ring))

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

    except Exception as e:
        logger.error(f"Error in ring classification for {sdf_file}: {e}")
        return [], [], []

def log_timeout(protein_file, ligand_file, start_time):
    """
    Log timeout information to a file.
    """
    try:
        with open(TIMEOUT_LOG_FILE, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration = time.time() - start_time
            f.write(f"[{timestamp}] TIMEOUT - Protein: {protein_file}, Ligand: {ligand_file}, Duration: {duration:.1f}s\n")
        logger.warning(f"Logged timeout for {ligand_file} to {TIMEOUT_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to log timeout: {e}")

def run_smina_docking_serial(args):
    """
    Run Smina docking with configurable CPU cores and timeout monitoring.
    """
    protein_file, ligand_file, autobox_ligand, output_dir, exhaustiveness = args
    start_time = time.time()
    
    try:
        protein_name = os.path.basename(protein_file).replace('_protonated.pdb', '')
        ligand_name = os.path.basename(ligand_file).replace('.sdf', '')
        output_file = os.path.join(output_dir, f"{protein_name}_{ligand_name}_docked.sdf")

        # Use configurable number of CPU cores per Smina job
        num_cpu_per_docking = CPU_CORES_PER_SMINA
        
        cmd = [
            'smina', '-r', protein_file, '-l', ligand_file,
            '--autobox_ligand', autobox_ligand, '-o', output_file,
            '--exhaustiveness', str(exhaustiveness), '--seed', '1',
            '--num_modes', '100', '--energy_range', '5', '--scoring', 'vinardo',
            '--cpu', str(num_cpu_per_docking)  # Specify number of CPU cores
        ]

        logger.info(f"Running Smina docking with {num_cpu_per_docking} cores: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)

        duration = time.time() - start_time
        if result.returncode == 0:
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"Smina docking completed: {output_file} (Duration: {duration:.1f}s)")
                return output_file
            else:
                logger.error(f"Smina output file not created or empty for {ligand_file} (Duration: {duration:.1f}s)")
                # Log the timeout/skip
                log_timeout(protein_file, ligand_file, start_time)
                return None
        else:
            logger.error(f"Smina error for {ligand_file}: {result.stderr} (Duration: {duration:.1f}s)")
            # Log the error
            log_timeout(protein_file, ligand_file, start_time)
            return None

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Smina timeout for {ligand_file} after {duration:.1f}s")
        # Log the timeout
        log_timeout(protein_file, ligand_file, start_time)
        return None
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error running Smina for {ligand_file}: {e} (Duration: {duration:.1f}s)")
        # Log the error
        log_timeout(protein_file, ligand_file, start_time)
        return None

def find_ligand_sdf_files(directory):
    """Find all ligand SDF files in the given directory."""
    ligand_pattern = os.path.join(directory, "*_ligand.sdf")
    return glob.glob(ligand_pattern)

def setup_environment():
    """Check if necessary tools (OpenBabel, Smina) are available."""
    try:
        result = subprocess.run(['obabel', '-V'], capture_output=True, text=True, check=True)
        logger.info(f"OpenBabel available: {result.stdout.strip()}")
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

def collect_all_docking_tasks(protonated_files, reference_ring_dir):
    """
    Collect ALL docking tasks from ALL proteins upfront to ensure maximum parallelization.
    """
    all_docking_tasks = []
    
    logger.info("Collecting all docking tasks from all proteins...")
    
    for protonated_file in tqdm(protonated_files, desc="Scanning proteins"):
        protein_dir = os.path.dirname(protonated_file)
        ligand_files = find_ligand_sdf_files(protein_dir)
        
        if not ligand_files:
            logger.warning(f"No ligand files found in {protein_dir}")
            continue

        autobox_ligand = ligand_files[0]
        
        # Collect ring matches for all ligands in this protein
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
            continue

        # Create docking tasks for all ring matches
        for ring_match in all_ring_matches:
            ring_type, atom_indices, ref_name, ref_path = ring_match
            all_docking_tasks.append((
                protonated_file,
                ref_path,
                autobox_ligand,
                protein_dir,
                30  # Lower exhaustiveness for faster processing
            ))
    
    logger.info(f"Collected {len(all_docking_tasks)} total docking tasks from {len(protonated_files)} proteins")
    return all_docking_tasks

def main_workflow():
    """
    Main workflow: Run ALL docking tasks in parallel to maximize CPU utilization.
    """
    logger.info("Starting automated molecular docking workflow...")
    logger.info(f"Configuration: {CONCURRENT_PROTEINS} concurrent proteins, {CPU_CORES_PER_SMINA} CPU cores per Smina job")
    logger.info(f"Timeout setting: {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS/60:.1f} minutes)")
    logger.info(f"Timeout log file: {TIMEOUT_LOG_FILE}")
    
    total_cores = cpu_count()
    max_possible_concurrent = total_cores // CPU_CORES_PER_SMINA
    logger.info(f"System has {total_cores} CPU cores")
    logger.info(f"Maximum possible concurrent Smina jobs: {max_possible_concurrent}")
    
    resources = get_system_resources()
    logger.info(f"System resources - CPU: {resources['cpu_percent']}%, "
                f"Memory: {resources['memory_percent']}% ({resources['memory_available_gb']:.1f}GB available)")

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

    # Step 2: Collect ALL docking tasks upfront
    logger.info("Step 2: Collecting all docking tasks")
    all_docking_tasks = collect_all_docking_tasks(protonated_files, reference_ring_dir)
    
    if not all_docking_tasks:
        logger.error("No docking tasks found. Exiting.")
        return
    
    logger.info(f"Found {len(all_docking_tasks)} total docking tasks to run")

    # Step 3: Run ALL docking tasks in parallel (maximum parallelization)
    logger.info("Step 3: Running all docking tasks in parallel")
    
    # Calculate optimal number of concurrent processes
    num_concurrent_docking = min(len(all_docking_tasks), max_possible_concurrent, 64)  # Cap at 64 to prevent overload
    
    logger.info(f"Running {len(all_docking_tasks)} docking tasks with {num_concurrent_docking} concurrent processes")
    logger.info(f"Each Smina job will use {CPU_CORES_PER_SMINA} CPU cores")
    logger.info(f"Expected maximum CPU utilization: {num_concurrent_docking * CPU_CORES_PER_SMINA} cores")
    logger.info(f"Timeout for individual jobs: {TIMEOUT_SECONDS} seconds")
    
    # Initialize timeout log file
    with open(TIMEOUT_LOG_FILE, 'w') as f:
        f.write(f"Smina Timeout Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    # Run ALL docking tasks in parallel
    with Pool(processes=num_concurrent_docking, maxtasksperchild=2) as pool:
        results = list(tqdm(
            pool.imap(run_smina_docking_serial, all_docking_tasks, chunksize=1),
            total=len(all_docking_tasks),
            desc="Running docking jobs",
            unit="docking"
        ))

    successful_results = [r for r in results if r is not None]
    failed_results = len(all_docking_tasks) - len(successful_results)
    
    logger.info(f"Completed {len(successful_results)} out of {len(all_docking_tasks)} docking tasks successfully")
    logger.info(f"Failed/skipped tasks: {failed_results}")
    
    # Report timeout statistics
    if os.path.exists(TIMEOUT_LOG_FILE):
        timeout_count = 0
        try:
            with open(TIMEOUT_LOG_FILE, 'r') as f:
                timeout_count = len([line for line in f if 'TIMEOUT' in line])
            if timeout_count > 0:
                logger.info(f"Total timeout/skipped jobs logged: {timeout_count}")
                logger.info(f"See {TIMEOUT_LOG_FILE} for details")
        except Exception as e:
            logger.error(f"Could not read timeout log: {e}")
    
    final_resources = get_system_resources()
    logger.info(f"Final system resources - CPU: {final_resources['cpu_percent']}%, "
                f"Memory: {final_resources['memory_percent']}%")
    
    logger.info("Automated molecular docking workflow completed!")

def get_system_resources():
    """Get current system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3)
    }

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main_workflow()
