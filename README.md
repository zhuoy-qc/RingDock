🧬 RingDock 
This repository provides tools for our RingDock pipepline, focusing on pi-cation interactions (cations from protein interacts with ligand aromatic rings).

⚙️ Environment Setup 
We recommend using Conda to set up the environment.

conda env create -f ringdock_pi-cation_env.yml

conda activate ringdock_pi-cation_env

📁 Preparation 
Download the ring_sdf_files dir and set the path in the codes to match this dir path

📁 Dataset-Based Usage (e.g., PoseBuster)
To run the full pipeline on a dataset like PoseBuster:

Download the dataset of interest.

Go to the single working directory containing all PDB ID dirs.

Run the following scripts in that directory:

python pi-cation-analysis.py, which finds all pi-cation interactions 
python 1_sampling.py
python 2_model.py




🧪 Single PDB Usage
You can also analyze individual PDB files using the utilities provided:

Navigate to the utils/ directory.



📊 Output & Evaluation
The analysis codes include:

Compute Recovery rate for π-cation interactions

Structural error metrics


