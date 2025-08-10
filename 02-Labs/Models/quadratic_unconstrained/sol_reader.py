import re
from typing import Tuple, Dict, Any


def parse_sol_file(
    file_path: str, n: int
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[int, float]]:
    """
    Parse a .sol-style file with headers and a sequence of values.

    Expected format:
      # Energy: 7
      # entries: 1113
      1
      0
      1
      0
      ...

    Returns:
      energy_dict   {"Energy": <float>}
      entries_dict  {"entries": <int>}
      sequence_dict {0: <float>, 1: <float>, …}
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
    solution_dict = {i: val for i, val in enumerate(sequence)}

    return energy_dict, entries_dict, solution_dict
