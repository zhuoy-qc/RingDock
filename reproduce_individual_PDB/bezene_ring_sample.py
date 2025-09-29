#!/usr/bin/env python3
#bezene are used as an example，should replace with other if needed
import os
import sys
from Bio.PDB import PDBParser, PDBIO, Select, PDBList
from rdkit import Chem

# -------------------- Selectors --------------------

class CleanPDBSelector(Select):
    def accept_residue(self, residue):
        return residue.resname.strip() not in ["HOH", "WAT", "CL", "NA", "K", "MG", "CA"]

class LigandSelector(Select):
    def __init__(self):
        self.valid_ligands = []

    def accept_residue(self, residue):
        if residue.id[0] != " ":
            temp_file = f"temp_{residue.resname}_{residue.id[1]}.pdb"
            io = PDBIO()
            io.set_structure(residue)
            io.save(temp_file)

            mol = Chem.MolFromPDBFile(temp_file)
            os.remove(temp_file)

            if mol and mol.GetNumAtoms() >= 9:
                self.valid_ligands.append({
                    'resname': residue.resname,
                    'chain': residue.parent.id,
                    'resnum': residue.id[1],
                    'atom_count': mol.GetNumAtoms(),
                    'residue': residue
                })
        return False

# -------------------- Core Functions --------------------

def download_pdb(pdb_id):
    pdb_file = f"{pdb_id}.pdb"
    if not os.path.exists(pdb_file):
        print(f"Downloading PDB file for {pdb_id}...")
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=".")
        downloaded_file = f"pdb{pdb_id.lower()}.ent"
        if os.path.exists(downloaded_file):
            os.rename(downloaded_file, pdb_file)
        else:
            print(f"Error: Downloaded file {downloaded_file} not found.")
            sys.exit(1)
    return pdb_file

def process_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", pdb_file)

    io = PDBIO()
    io.set_structure(structure)
    clean_file = f"{os.path.splitext(pdb_file)[0]}_protein.pdb"
    io.save(clean_file, select=CleanPDBSelector())
    print(f"\nSaved cleaned protein to {clean_file}")

    ligand_selector = LigandSelector()
    io.set_structure(structure)
    io.save("temp_ligand_check.pdb", select=ligand_selector)
    os.remove("temp_ligand_check.pdb")

    if not ligand_selector.valid_ligands:
        print("Warning: No ligands with ≥9 atoms found")
        return None, None

    return clean_file, ligand_selector.valid_ligands

def convert_to_sdf_and_protonate(ligand_residue, resname_tag, output_dir="."):
    temp_pdb = os.path.join(output_dir, f"temp_{resname_tag}.pdb")
    io = PDBIO()
    io.set_structure(ligand_residue)
    io.save(temp_pdb)

    mol = Chem.MolFromPDBFile(temp_pdb)
    os.remove(temp_pdb)

    if not mol:
        print(f"Error: Could not parse ligand {resname_tag}")
        return None, None

    sdf_file = os.path.join(output_dir, f"ligand_{resname_tag.replace('-', '_')}.sdf")
    protonated_sdf = os.path.join(output_dir, f"ligand_{resname_tag.replace('-', '_')}_H.sdf")

    writer = Chem.SDWriter(sdf_file)
    writer.write(mol)
    writer.close()
    print(f"Converted {resname_tag} to SDF format: {sdf_file}")

    os.system(f"obabel {sdf_file} -O {protonated_sdf} -p 2>/dev/null")
    if os.path.exists(protonated_sdf):
        print(f"Protonated ligand saved to: {protonated_sdf}")
    else:
        print("Error: Failed to protonate ligand using obabel")
        protonated_sdf = None

    return sdf_file, protonated_sdf

def protonate_protein(pdb_file, ligand_info, output_dir="."):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein_with_other_ligands", pdb_file)

    class ProteinWithOtherLigandsSelector(Select):
        def accept_residue(self, residue):
            if residue.id[0] == " ":
                return True
            if (residue.resname == ligand_info['resname'] and
                residue.parent.id == ligand_info['chain'] and
                residue.id[1] == ligand_info['resnum']):
                return False
            return True

    temp_file = os.path.join(output_dir, "temp_protein_with_other_ligands.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(temp_file, select=ProteinWithOtherLigandsSelector())

    protonated_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdb_file))[0]}_H.pdb")
    os.system(f"obabel {temp_file} -O {protonated_file} -p 2>/dev/null")
    os.remove(temp_file)

    if os.path.exists(protonated_file):
        print(f"Protonated protein saved to {protonated_file}")
        return protonated_file
    else:
        print("Error: Failed to protonate protein using obabel")
        return None

def create_bz_sdf(output_dir="."):
    bz_sdf_content = """241
  -OEChem-09092502143D

 12 12  0     0  0  0  0  0  0999 V2000
   -1.2131   -0.6884    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2028    0.7064    0.0001 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0103   -1.3948    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0104    1.3948   -0.0001 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2028   -0.7063    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2131    0.6884    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1577   -1.2244    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1393    1.2564    0.0001 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0184   -2.4809   -0.0001 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0184    2.4808    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1394   -1.2563    0.0001 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1577    1.2245    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  1  3  1  0  0  0  0
  1  7  1  0  0  0  0
  2  4  1  0  0  0  0
  2  8  1  0  0  0  0
  3  5  2  0  0  0  0
  3  9  1  0  0  0  0
  4  6  2  0  0  0  0
  4 10  1  0  0  0  0
  5  6  1  0  0  0  0
  5 11  1  0  0  0  0
  6 12  1  0  0  0  0
M  END
$$$$"""

    bz_sdf_path = os.path.join(output_dir, "bz.sdf")
    with open(bz_sdf_path, 'w') as f:
        f.write(bz_sdf_content)
    print(f"\nCreated bz.sdf in {output_dir}")
    return bz_sdf_path

def run_smina_docking(protein_file, ligand_file, autobox_ligand, output_dir="."):
    output_file = os.path.join(output_dir, "docked_ring_poses.sdf")
    cmd = (
        f"smina -r {protein_file} -l {ligand_file} "
        f"--autobox_ligand {autobox_ligand} -o {output_file} "
        "--exhaustiveness 80 --seed 1 --scoring vinardo --num_modes 100 --energy_range 5 --min_rmsd_filter 1"
    )
    print(f"\nRunning docking command:\n{cmd}")
    os.system(cmd)
    
    if os.path.exists(output_file):
        print(f"\nDocking results saved to {output_file}")
    else:
        print("\nError: Docking failed")

# -------------------- Main Execution --------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_docking.py <PDB_ID>")
        sys.exit(1)

    pdb_id = sys.argv[1].upper()
    pdb_file = download_pdb(pdb_id)

    output_dir = f"{pdb_id}_prepared"
    os.makedirs(output_dir, exist_ok=True)

    protein_file, ligands = process_pdb(pdb_file)
    if not ligands:
        print("No valid ligands found. Exiting.")
        sys.exit(1)

    new_protein_file = os.path.join(output_dir, os.path.basename(protein_file))
    os.rename(protein_file, new_protein_file)
    protein_file = new_protein_file

    print("\nAvailable ligands:")
    for idx, ligand in enumerate(ligands):
        print(f"{idx+1}. {ligand['resname']}-{ligand['chain']}-{ligand['resnum']} ({ligand['atom_count']} atoms)")

    while True:
        try:
            choice = int(input(f"\nEnter the number of the ligand to process (1-{len(ligands)}): "))
            if 1 <= choice <= len(ligands):
                selected_ligand = ligands[choice - 1]
                break
            else:
                print("Please enter a valid number")
        except ValueError:
            print("Please enter a number")

    resname_tag = f"{selected_ligand['resname']}-{selected_ligand['chain']}-{selected_ligand['resnum']}"
    ligand_sdf, ligand_sdf_H = convert_to_sdf_and_protonate(selected_ligand['residue'], resname_tag, output_dir)
    protonated_protein = protonate_protein(protein_file, selected_ligand, output_dir)

    print("\nProcessing complete!")
    print(f"\nFiles generated in '{output_dir}':")
    print(f"- Cleaned protein: {os.path.basename(protein_file)}")
    print(f"- Protonated protein: {os.path.basename(protonated_protein) if protonated_protein else '[FAILED]'}")
    print(f"- Ligand in SDF format: {os.path.basename(ligand_sdf) if ligand_sdf else '[FAILED]'}")
    print(f"- Protonated ligand: {os.path.basename(ligand_sdf_H) if ligand_sdf_H else '[FAILED]'}")

    # Run smina docking if all required files exist
    if protonated_protein and ligand_sdf_H:
        # Create bz.sdf in the output directory
        bz_sdf = create_bz_sdf(output_dir)
        
        run_smina_docking(
            protein_file=protonated_protein,
            ligand_file=bz_sdf,
            autobox_ligand=ligand_sdf_H,
            output_dir=output_dir
        )
    else:
        print("\nCannot run docking - missing required files")

if __name__ == "__main__":
    main()
