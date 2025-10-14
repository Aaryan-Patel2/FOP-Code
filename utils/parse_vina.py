import re
import os

def parse_vina_score(pdbqt_file):
    """
    Extract the docking score from a Vina output PDBQT file.
    If the file doesn't exist but an alternative format (e.g., .sdf) exists,
    attempts to find and parse that instead.
    """
    # Check if the file exists
    if not os.path.exists(pdbqt_file):
        # Try alternative file extensions
        base_name = os.path.splitext(pdbqt_file)[0]
        alternative_extensions = ['.sdf', '.pdb', '.mol2']
        
        for ext in alternative_extensions:
            alt_file = base_name + ext
            if os.path.exists(alt_file):
                print(f"INFO: Using {alt_file} instead of {pdbqt_file}")
                pdbqt_file = alt_file
                break
        else:
            print(f"ERROR: File {pdbqt_file} not found and no alternatives found")
            return 0.0
    
    try:
        with open(pdbqt_file, 'r') as f:
            content = f.read()
        
        # Look for the affinity score in the REMARK lines
        # Example: "REMARK VINA RESULT: -9.1 0.000 0.000"
        match = re.search(r'REMARK\s+VINA\s+RESULT:\s+([-\d\.]+)', content)
        if match:
            return float(match.group(1))
        else:
            # Alternative pattern some Vina versions use
            match = re.search(r'REMARK\s+Affinity:\s+([-\d\.]+)', content)
            if match:
                return float(match.group(1))
            else:
                print(f"WARNING: No score found in {pdbqt_file}")
                return 0.0  # Default neutral score
    except FileNotFoundError:
        print(f"ERROR: File {pdbqt_file} not found")
        return 0.0
    except Exception as e:
        print(f"ERROR: Failed to parse {pdbqt_file}: {e}")
        return 0.0

# Alternative: Parse from log files if you saved them
def parse_vina_log(log_file):
    """
    Extract scores from Vina log files
    """
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "Affinity" in line and "kcal/mol" in line:
                    # Example: "Affinity: -8.1 kcal/mol"
                    return float(line.split()[1])
        return 0.0
    except:
        return 0.0