#!/usr/bin/env python3
import os
import sys
import glob
import csv
import math
import numpy as np
import pandas as pd
import joblib
from decimal import Decimal
from multiprocessing import Pool, cpu_count, set_start_method
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from plip.structure.preparation import PDBComplex

# --- Step 1: Clean PDB to retain only standard amino acids ---
class ProteinSelect(Select):
    def accept_model(self, model): return True
    def accept_chain(self, chain): return True
    def accept_residue(self, residue): return is_aa(residue, standard=True)

def clean_pdb(input_file, output_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_file)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, ProteinSelect())
    print(f"✅ Clean protein structure saved to {output_file}")

# --- Step 2: Generate protein-ligand complexes ---
def find_protein_file():
    protein_files = glob.glob("*_protein_clean.pdb")
    if not protein_files:
        raise FileNotFoundError("❌ No *_protein_clean.pdb file found")
    if len(protein_files) > 1:
        print(f"  Multiple protein files found: {protein_files}")
        print(f"Using the first one: {protein_files[0]}")
    return protein_files[0]

def get_user_input():
    """Use default SDF filename without prompting the user"""
    sdf_filename = "docked_ring_poses"
    full_path = f"{sdf_filename}.sdf"
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ File not found: {full_path}")
    return full_path

def create_complexes_direct(protein_file, multi_pose_sdf, output_prefix, output_dir="run_plip", num_poses=None):
    os.makedirs(output_dir, exist_ok=True)
    supplier = Chem.SDMolSupplier(multi_pose_sdf)
    successful_complexes = []

    for i, mol in enumerate(supplier):
        if mol is None:
            print(f"  Skipping unreadable molecule {i+1}")
            continue
        if num_poses and i >= num_poses:
            break

        output_file = os.path.join(output_dir, f"{output_prefix}{i+1}_complex.pdb")
        try:
            with open(protein_file, 'r') as protein, open(output_file, 'w') as output:
                for line in protein:
                    if line.startswith(('ATOM', 'HETATM')):
                        output.write(line)
                output.write(Chem.MolToPDBBlock(mol))
                output.write("END\n")
            successful_complexes.append(output_file)
            print(f"✅ Complex created: {output_file}")
        except Exception as e:
            print(f"❌ Error creating complex {i+1}: {e}")
    return successful_complexes

# --- Step 3: π-Cation Interaction Analysis ---
def calculate_angle(ring_normal, charge_vector):
    dot_product = Decimal(ring_normal[0]) * Decimal(charge_vector[0]) + \
                  Decimal(ring_normal[1]) * Decimal(charge_vector[1]) + \
                  Decimal(ring_normal[2]) * Decimal(charge_vector[2])
    norm_ring = Decimal(math.sqrt(sum(x**2 for x in ring_normal)))
    norm_charge = Decimal(math.sqrt(sum(x**2 for x in charge_vector)))
    cos_theta = dot_product / (norm_ring * norm_charge)
    cos_theta = max(min(float(cos_theta), 1.0), -1.0)
    return math.degrees(math.acos(cos_theta))

def calculate_rz(distance, offset):
    try:
        rz_squared = float(distance)**2 - float(offset)**2
        return math.sqrt(max(rz_squared, 0))
    except (ValueError, TypeError):
        return float('nan')

def analyze_pication_interactions(pdb_file):
    my_mol = PDBComplex()
    my_mol.load_pdb(pdb_file)
    my_mol.analyze()
    results = []

    for _, interactions in my_mol.interaction_sets.items():
        for pication in interactions.all_pication_laro:
            ring_normal = np.array(pication.ring.normal, dtype=np.float64)
            charge_vector = np.array(pication.charge.center, dtype=np.float64) - np.array(pication.ring.center, dtype=np.float64)
            angle = calculate_angle(ring_normal, charge_vector)
            rz = calculate_rz(pication.distance, pication.offset)

            results.append({
                'PDB_File': pdb_file,
                'Ligand': f"{pication.restype_l}-{pication.resnr_l}-{pication.reschain_l}",
                'Protein': f"{pication.restype}-{pication.resnr}-{pication.reschain}",
                'Distance': format(float(pication.distance), '.16g'),
                'Offset': format(float(pication.offset), '.16g'),
                'RZ': format(rz, '.16g'),
                'Angle': format(angle, '.16g'),
                'Ring_Center_X': format(float(pication.ring.center[0]), '.16g'),
                'Ring_Center_Y': format(float(pication.ring.center[1]), '.16g'),
                'Ring_Center_Z': format(float(pication.ring.center[2]), '.16g'),
                'Charged_Center_X': format(float(pication.charge.center[0]), '.16g'),
                'Charged_Center_Y': format(float(pication.charge.center[1]), '.16g'),
                'Charged_Center_Z': format(float(pication.charge.center[2]), '.16g'),
                'Ring_Normal_X': format(float(ring_normal[0]), '.16g'),
                'Ring_Normal_Y': format(float(ring_normal[1]), '.16g'),
                'Ring_Normal_Z': format(float(ring_normal[2]), '.16g'),
                'Ring_Type': pication.ring.type,
                'Atom_Indices': str(pication.ring.atoms_orig_idx)
            })
    return results

def process_single_pdb(pdb_file):
    try:
        results = analyze_pication_interactions(pdb_file)
        print(f"✅ Processed {pdb_file}: {len(results)} π-cation interactions")
        return results
    except Exception as e:
        print(f"❌ Error processing {pdb_file}: {str(e)}")
        return []

def process_all_pdbs_parallel():
    pdb_files = [f for f in os.listdir('run_plip') if f.endswith('.pdb') and 'complex' in f.lower()]
    if not pdb_files:
        print("❌ No PDB files found in 'run_plip' directory!")
        return

    print(f"🔍 Found {len(pdb_files)} PDB files to process")
    num_processes = min(90, cpu_count())
    print(f"  Using {num_processes} CPU cores")

    all_results = []
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_pdb, [os.path.join('run_plip', f) for f in pdb_files])
        for result in results:
            all_results.extend(result)

    if all_results:
        csv_file = 'pication_interaction_analysis_all.csv'
        fieldnames = list(all_results[0].keys())
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n📁 Saved all results to {csv_file}")
        print(f"📊 Total interactions found: {len(all_results)}")
        return csv_file
    else:
        print("\n  No π-cation interactions found.")
        return None

# --- Step 4: Model Prediction and Ranking ---
def run_model_prediction(input_csv_path):
    """Use machine learning model for prediction and ranking"""
    # Path configuration
    model_path = '/home/zyin/final_model_20250919_025353.pkl'  # Change this to ensure model path are correct !!!!!!!
    output_csv_path = 'predictions_with_energy_ranked.csv'

    # 1. Load trained model
    try:
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Read CSV file
    try:
        df = pd.read_csv(input_csv_path)
        print(f"✅ Successfully read CSV file: {input_csv_path}")
        print(f"📊 Data shape: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to read CSV file: {e}")
        return

    # 3. Column mapping and feature extraction
    required_columns_in_csv = ['Offset', 'RZ', 'Angle', 'Distance']
    if not all(col in df.columns for col in required_columns_in_csv):
        missing_cols = [col for col in required_columns_in_csv if col not in df.columns]
        print(f"❌ CSV file is missing required columns: {missing_cols}")
        print("Please ensure CSV file contains: ['Offset', 'RZ', 'Angle', 'Distance']")
        return

    # Extract features from raw data and rename to match model expectations
    feature_mapping = {
        'Offset': 'delta_x',
        'RZ': 'delta_z',
        'Angle': 'dihedral_angle',
        'Distance': 'distance'
    }

    # Create model input feature DataFrame
    X_input = df[required_columns_in_csv].copy()
    X_input.rename(columns=feature_mapping, inplace=True)

    print("\n🔧 Model input features preview (after mapping):")
    print(X_input.head())

    # 4. Data preprocessing (handle missing values)
    if X_input.isnull().any().any():
        print("⚠️  Input data contains missing values, processing...")
        X_input_clean = X_input.dropna()
        print(f"📊 Cleaned feature data shape: {X_input_clean.shape}")
    else:
        X_input_clean = X_input
        print("✅ Input data contains no missing values.")

    # 5. Make predictions using the model
    try:
        print("\n🔮 Making energy predictions...")
        # Ensure feature order matches training
        expected_feature_order = ['delta_z', 'delta_x', 'dihedral_angle', 'distance']
        X_input_clean = X_input_clean[expected_feature_order]
        predicted_energies = model.predict(X_input_clean)
        print("✅ Prediction completed!")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return

    # 6. Add predictions back to original DataFrame
    df_pred = df.loc[X_input_clean.index].copy()  # Keep only rows used for prediction
    df_pred['Predicted_Energy'] = predicted_energies

    # 7. Sort by predicted energy (ascending, lower energy is more stable)
    df_pred_sorted = df_pred.sort_values(by='Predicted_Energy', ascending=True)
    # Reset index and add ranking column
    df_pred_sorted.reset_index(drop=True, inplace=True)
    df_pred_sorted['Energy_Rank'] = df_pred_sorted.index + 1

    # 8. Save results to new CSV file
    try:
        df_pred_sorted.to_csv(output_csv_path, index=False)
        print(f"💾 Results saved to: {output_csv_path}")
    except Exception as e:
        print(f"❌ Failed to save results file: {e}")

    # 9. Display top 10 predictions (sorted by energy)
    print("\n🏆 Top 10 complexes (sorted by predicted energy, lower is more stable):")
    cols_to_display = ['PDB_File', 'Ligand', 'Protein', 'Distance', 'Offset', 'RZ', 'Angle', 'Predicted_Energy', 'Energy_Rank']
    print(df_pred_sorted.head(10)[cols_to_display].to_string(index=False))

    print(f"\n🎉 Processing complete! Ranked results saved to {output_csv_path}")
    return output_csv_path

# --- Main Execution ---
if __name__ == "__main__":
    set_start_method('fork', force=True)

    print("=" * 60)
    print("🔬 π-CATION INTERACTION ANALYSIS & ENERGY PREDICTION")
    print("=" * 60)
    
    # Step 1: Clean raw PDB
    print("\n1️⃣  STEP 1: CLEANING PROTEIN STRUCTURE")
    print("-" * 40)
    raw_pdb_files = glob.glob("*_protein.pdb")
    if not raw_pdb_files:
        print("❌ No *_protein.pdb file found")
        sys.exit(1)

    raw_pdb = raw_pdb_files[0]
    clean_pdb(raw_pdb, f"{os.path.splitext(raw_pdb)[0]}_clean.pdb")

    # Step 2: Generate complexes
    print("\n2️⃣  STEP 2: GENERATING PROTEIN-LIGAND COMPLEXES")
    print("-" * 40)
    protein_file = find_protein_file()
    sdf_file = get_user_input()
    create_complexes_direct(protein_file, sdf_file, output_prefix="dock", output_dir="run_plip")

    # Step 3: Analyze interactions
    print("\n3️⃣  STEP 3: ANALYZING π-CATION INTERACTIONS")
    print("-" * 40)
    csv_file = process_all_pdbs_parallel()

    # Step 4: Model prediction and ranking
    if csv_file:
        print("\n4️⃣  STEP 4: RUNNING MACHINE LEARNING PREDICTION")
        print("-" * 40)
        run_model_prediction(csv_file)
    
    print("\n" + "=" * 60)
    print("🎯 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
