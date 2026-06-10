"""Fixed dimensions shared by model.py, dat_reader.py, and sol_reader.py.

The values come from parameter_u3_c10.zpl of the original QOBLIB repository:
the number of units per asset and position (ub), the position signs
(long/short), and the binary expansion widths of the two slack variables
(CS1, CS2).
"""

from typing import Final

NUM_UNITS: Final[int] = 3
NUM_SIGNS: Final[int] = 2
NUM_Y_SLACKS: Final[int] = 4
NUM_S_SLACKS: Final[int] = 7
