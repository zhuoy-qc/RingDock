🧬 RingDock Pi-Cation Interaction Analysis Toolkit
This repository provides tools for analyzing π-cation interactions in protein structures, with support for both large datasets (e.g., PoseBuster) and individual PDB files. The pipeline includes sampling, modeling, and evaluation of interaction recovery rates.

⚙️ Environment Setup
We recommend using Conda to set up a reproducible environment.

Step 1: Create the Environment
bash
conda env create -f ringdock_pi-cation_env.yml
💡 Before sharing the YAML file, remove the prefix: line to avoid machine-specific paths.

Step 2: Activate the Environment
bash
conda activate ringdock_pi-cation_env
📁 Dataset-Based Usage (e.g., PoseBuster)
To run the full pipeline on a dataset like PoseBuster:

Download the dataset of interest.

Place all PDB_ID directories in a single working directory.

Run the following scripts in that directory:

python pi-cation-analysis.py
python 1_sampling.py
python 2_model.py
The toolkit will compute:

π-cation interaction recovery rates

Error metrics for structural predictions

🧪 Single PDB Usage
You can also analyze individual PDB files using the utilities provided:

Navigate to the utils/ directory.

Use the scripts there to run the pipeline on a single structure.

This mode is ideal for quick testing or focused analysis on specific proteins.

📊 Output & Evaluation
The analysis outputs include:

Recovery statistics for π-cation interactions

Structural error metrics

Logs and intermediate data for further inspection
