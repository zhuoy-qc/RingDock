we suggest use conda to set up a new  enviroment
conda env create -f ringdock_pi-cation_env.yml


Example usage on the dataset such as Posebuster:
First, downolad the dataset you interested, run pi-cation-analysis.py in the dir containing all the PDB_ID dirs
Run 1_sampling and the 2_model.py in the dir containing all the PDB_ID dirs with pi-cation interactions
Analysis using the provided code which computes errors and pi-cation interaction recovery rate 



One can also use this code on any single PDB easily, see the attached utils/codes for run for a single PDB.
