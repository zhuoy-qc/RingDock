import os
import shutil

def move_directories_with_pication_posebuster():
    # Create the target directory if it doesn't exist
    target_dir = "with_pication"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    # Read the input file
    try:
        with open("PBD_lists_withpi-cation-interactions.txt", "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("Error: 'directories_with_interactions.txt' not found in current directory")
        return

    # Process each line
    for line in lines:
        # Remove whitespace and newline characters
        dir_name = line.strip()

        # Skip empty lines
        if not dir_name:
            continue

        # Check if the directory exists in current directory
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            try:
                # Move the directory to the target directory
                shutil.move(dir_name, os.path.join(target_dir, dir_name))
                print(f"Moved: {dir_name} -> {target_dir}/{dir_name}")
            except Exception as e:
                print(f"Error moving {dir_name}: {e}")
        else:
            print(f"Directory not found: {dir_name}")

if __name__ == "__main__":
    move_directories_with_pication_posebuster()
