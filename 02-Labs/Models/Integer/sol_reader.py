import re
from typing import Tuple, Dict


def parse_sol_file(
    file_path: str, n: int
) -> Tuple[Dict[str, float], Dict[str, int], Dict[int, float]]:
    """
    Parses a `.sol` solution file to extract the optimization result energy,
    the number of consecutive entries, and the variable assignment sequence.

    Args:
        file_path (str): 
            Path to the `.sol` solution file containing header metadata and
            numeric variable assignments.
        n (int): 
            Starting index offset for variable IDs. The variable IDs in the
            output `solution_dict` will start from `n - 1`.

    Returns:
        Tuple[Dict[str, float], Dict[str, int], Dict[int, float]]:
            - energy_dict: A dictionary with the key `"Energy"` mapped to the 
              parsed energy value from the file.
            - entries_dict: A dictionary with the key `"entries"` mapped to 
              the parsed number of consecutive entries.
            - solution_dict: A mapping from variable ID (starting at `n - 1`) 
              to its assigned float value, based on the sequence in the file.
    """
    energy = None
    entries = None
    sequence = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header lines
            if line.startswith("#"):
                # match Energy
                m = re.match(r"#\s*Energy\s*:\s*([-+]?\d*\.?\d+)", line, re.IGNORECASE)
                if m:
                    energy = float(m.group(1))
                    continue

                # match entries
                m = re.match(
                    r"#\s*Consecutive entries\s*:\s*(\d+)", line, re.IGNORECASE
                )
                if m:
                    entries = int(m.group(1))
                    continue

            # Numeric lines → collect into sequence
            try:
                sequence.append(float(line))
            except ValueError:
                # skip any non-numeric lines
                continue

    # Build dicts
    energy_dict = {"Energy": energy}
    entries_dict = {"entries": entries}
    solution_dict = {i + n - 1: val for i, val in enumerate(sequence)}

    return energy_dict, entries_dict, solution_dict
