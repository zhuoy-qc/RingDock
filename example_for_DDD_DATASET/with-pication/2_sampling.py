import os
import subprocess
import logging
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import psutil
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURABLE PARAMETERS ===
CONCURRENT_PROTEINS = 12      # Number of proteins to process simultaneously
CPU_CORES_PER_SMINA = 8       # Number of CPU cores each protein job will use
TIMEOUT_SECONDS = 120         # Timeout in seconds (2 minutes)
EXHAUSTIVENESS = 100          # Exhaustiveness parameter for Smina, larger means more time to sample, suggested value no larger than 100
# ===============================

# Global variable for timeout log file
TIMEOUT_LOG_FILE = "smina_timeouts.log"

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
    Run Smina sampling with configurable CPU cores and timeout monitoring.
    """
    protein_file, ligand_file, autobox_ligand, output_dir, exhaustiveness = args
    start_time = time.time()

    try:
        protein_name = os.path.basename(protein_file).replace('_protonated.pdb', '')
        ligand_name = os.path.basename(ligand_file).replace('.sdf', '')
        output_file = os.path.join(output_dir, f"{protein_name}_{ligand_name}_docked.sdf")

        # Use configurable number of CPU cores per Smina job
        num_cpu_per_sampling = CPU_CORES_PER_SMINA

        cmd = [
            'smina', '-r', protein_file, '-l', ligand_file,
            '--autobox_ligand', autobox_ligand, '-o', output_file,
            '--exhaustiveness', str(exhaustiveness), '--seed', '1',
            '--num_modes', '150', '--energy_range', '20', '--scoring', 'vinardo',
            '--cpu', str(num_cpu_per_sampling)  
        ]

        logger.info(f"Running Smina sampling with {num_cpu_per_sampling} cores: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)

        duration = time.time() - start_time
        if result.returncode == 0:
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"Smina sampling completed: {output_file} (Duration: {duration:.1f}s)")
                return output_file
            else:
                logger.error(f"Smina output file not created or empty for {ligand_file} (Duration: {duration:.1f}s)")
                log_timeout(protein_file, ligand_file, start_time)
                return None
        else:
            logger.error(f"Smina error for {ligand_file}: {result.stderr} (Duration: {duration:.1f}s)")
            log_timeout(protein_file, ligand_file, start_time)
            return None

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Smina timeout for {ligand_file} after {duration:.1f}s")
        log_timeout(protein_file, ligand_file, start_time)
        return None
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error running Smina for {ligand_file}: {e} (Duration: {duration:.1f}s)")
        log_timeout(protein_file, ligand_file, start_time)
        return None

def setup_environment():
    """Check if necessary tools (Smina) are available."""
    try:
        subprocess.run(['smina', '--help'], capture_output=True, check=True)
        logger.info("Smina is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Smina is not installed or not in PATH")
        return False

    return True

def get_system_resources():
    """Get current system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3)
    }

def main_sampling():
    """
    Main sampling workflow: Load tasks and run Smina sampling.
    """
    logger.info("Starting Smina sampling workflow...")
    logger.info(f"Configuration: {CONCURRENT_PROTEINS} concurrent proteins, {CPU_CORES_PER_SMINA} CPU cores per Smina job")
    logger.info(f"Exhaustiveness setting: {EXHAUSTIVENESS}")
    logger.info(f"Timeout setting: {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS/60:.1f} minutes)")
    logger.info(f"Timeout log file: {TIMEOUT_LOG_FILE}")

    # Load sampling tasks
    tasks_file = "docking_tasks.pkl"
    if not os.path.exists(tasks_file):
        logger.error(f"Sampling tasks file not found: {tasks_file}")
        logger.error("Please run prepare_docking_tasks.py first!")
        return

    with open(tasks_file, 'rb') as f:
        all_sampling_tasks = pickle.load(f)

    # Update exhaustiveness in all tasks
    updated_tasks = []
    for task in all_sampling_tasks:
        protein_file, ligand_file, autobox_ligand, output_dir, _ = task
        updated_tasks.append((protein_file, ligand_file, autobox_ligand, output_dir, EXHAUSTIVENESS))
    
    all_sampling_tasks = updated_tasks

    logger.info(f"Loaded {len(all_sampling_tasks)} sampling tasks from {tasks_file}")

    total_cores = cpu_count()
    max_possible_concurrent = total_cores // CPU_CORES_PER_SMINA
    logger.info(f"System has {total_cores} CPU cores")
    logger.info(f"Maximum possible concurrent Smina jobs: {max_possible_concurrent}")
    
    resources = get_system_resources()
    logger.info(f"System resources - CPU: {resources['cpu_percent']}%, "
                f"Memory: {resources['memory_percent']}% ({resources['memory_available_gb']:.1f}GB available)")
    
    if not setup_environment():
        logger.error("Environment setup failed. Please install required tools.")
        return

    # Calculate optimal number of concurrent processes
    num_concurrent_sampling = min(len(all_sampling_tasks), max_possible_concurrent, 64)  # Cap at 64 to prevent overload

    logger.info(f"Running {len(all_sampling_tasks)} sampling tasks with {num_concurrent_sampling} concurrent processes")
    logger.info(f"Each Smina job will use {CPU_CORES_PER_SMINA} CPU cores")
    logger.info(f"Expected maximum CPU utilization: {num_concurrent_sampling * CPU_CORES_PER_SMINA} cores")
    logger.info(f"Timeout for individual jobs: {TIMEOUT_SECONDS} seconds")        
    # Initialize timeout log file
    with open(TIMEOUT_LOG_FILE, 'w') as f:
        f.write(f"Smina Timeout Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

    # Run ALL sampling tasks in parallel
    with Pool(processes=num_concurrent_sampling, maxtasksperchild=2) as pool:
        results = list(tqdm(
            pool.imap(run_smina_docking_serial, all_sampling_tasks, chunksize=1),
            total=len(all_sampling_tasks),
            desc="Running sampling jobs",
            unit="sampling"
        ))

    successful_results = [r for r in results if r is not None]
    failed_results = len(all_sampling_tasks) - len(successful_results)

    logger.info(f"Completed {len(successful_results)} out of {len(all_sampling_tasks)} sampling tasks successfully")
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

    logger.info("Smina sampling workflow completed!")

if __name__ == "__main__":
    main_sampling()

