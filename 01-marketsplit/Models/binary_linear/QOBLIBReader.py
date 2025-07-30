import numpy as np
from typing import Dict, List, Tuple, Any

class QOBLIBReader:
    """
    A class to read and parse QOBLIB-format .dat files, and convert them into a format
    usable by JijModeling (e.g., constraint matrix A, right-hand side vector b).
    """

    def __init__(self, filepath: str):
        """
        Initialize the reader with the path to a QOBLIB .dat file.

        Parameters:
        - filepath (str): Path to the .dat file.
        """
        self.filepath = filepath
        self.m = 0  # Number of constraints
        self.n = 0  # Number of variables
        self.A = None  # Coefficient matrix A (m x n)
        self.b = None  # Right-hand side vector b (length m)

    def read_dat_file(self) -> Dict[str, Any]:
        """
        Read and parse the QOBLIB .dat file.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'I': Index set for constraints (np.arange(m))
                - 'J': Index set for variables (np.arange(n))
                - 'a': Coefficient matrix A (shape: m x n)
                - 'b': Right-hand side vector b (length m)
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove comment lines and strip whitespace
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)

        if not lines:
            raise ValueError("No data lines found in the file.")

        # First line: contains m (number of constraints) and n (number of variables)
        first_line = lines[0].split()
        self.m = int(first_line[0])
        self.n = int(first_line[1])

        print(f"Parsed {self.m} constraints and {self.n} variables.")

        # Collect all remaining numbers from the file
        all_numbers = []
        for i in range(1, len(lines)):
            numbers = lines[i].split()
            all_numbers.extend([int(x) for x in numbers])

        # Validate total number of values
        expected_numbers = self.m * (self.n + 1)  # Each row: n coefficients + 1 constant
        if len(all_numbers) != expected_numbers:
            raise ValueError(f"Data length mismatch: expected {expected_numbers}, got {len(all_numbers)}")

        # Reshape into A (m x n) and b (length m)
        self.A = []
        self.b = []

        for i in range(self.m):
            start_idx = i * (self.n + 1)
            end_idx = start_idx + self.n + 1
            row_data = all_numbers[start_idx:end_idx]

            self.A.append(row_data[:-1])  # First n elements are coefficients
            self.b.append(row_data[-1])   # Last element is the RHS constant

        self.A = np.array(self.A)
        self.b = np.array(self.b)

        return {
            'I': np.arange(self.m),
            'J': np.arange(self.n),
            'a': self.A,
            'b': self.b
        }