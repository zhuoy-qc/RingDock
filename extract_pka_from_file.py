def extract_pka_value(filename, residue_name, residue_number, chain_id):
    """
    Extracts the pKa value for a specific residue from a .pka file.

    Args:
        filename: Path to the .pka file
        residue_name: Name of the residue (e.g., 'HIS')
        residue_number: Number of the residue (e.g., 1057)
        chain_id: Chain identifier (e.g., 'A')

    Returns:
        The pKa value as a float, or None if not found
    """
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Skip header lines that don't start with residue info
                if line.strip() and not line.startswith('Group'):
                    parts = line.split()
                    if len(parts) >= 3:
                        group_info = parts[0] + ' ' + parts[1] + ' ' + parts[2]

                        # Extract residue name, number, and chain from the group info
                        # Format is like "HIS1057 A" where HIS1057 is residue name+number and A is chain
                        res_name_num = parts[0]
                        res_chain = parts[1]

                        # Extract the number from the residue name+number string
                        res_name = ''
                        res_num = ''
                        for i, char in enumerate(res_name_num):
                            if char.isdigit():
                                res_name = res_name_num[:i]
                                res_num = res_name_num[i:]
                                break

                        if (res_name == residue_name and
                            int(res_num) == residue_number and
                            res_chain == chain_id):
                            return float(parts[2])  # Return the pKa value

        # If we get here, the residue was not found
        print(f"Residue {residue_name}{residue_number} {chain_id} not found in {filename}")
        return None

    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Usage example for HIS-1057-A
filename = "7MMH_ZJY_only_protein.pka"
residue_name = "HIS"
residue_number = 1057
chain_id = "A"

pka_value = extract_pka_value(filename, residue_name, residue_number, chain_id)
if pka_value is not None:
    print(f"The pKa value for {residue_name}-{residue_number}-{chain_id} is: {pka_value}")
