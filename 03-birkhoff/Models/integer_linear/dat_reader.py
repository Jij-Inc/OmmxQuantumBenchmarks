import numpy as np
import math

def permutation_to_matrix(perm):
    n = len(perm)
    matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        matrix[i][perm[i]-1]=1
    return matrix
    
def convert_all_permutations(perms):
    return[permutation_to_matrix(p) for p in perms]

def read_permutation_dat_file(filepath):
    result = []
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                numbers = list(map(int, stripped.split()))
                result.append(numbers)
    return result

def process_entry(entry):
    n = entry["n"]
    scale = entry["scale"]

    A = np.array(entry["scaled_doubly_stochastic_matrix"]).reshape((n, n))
    if n == 3:
        perm_list = read_permutation_dat_file("p3.dat")
        P = convert_all_permutations(perm_list)
    elif n == 4:
        perm_list = read_permutation_dat_file("p4.dat")
        P = convert_all_permutations(perm_list)
    elif n == 5:
        perm_list = read_permutation_dat_file("p5.dat")
        P = convert_all_permutations(perm_list)
    elif n == 6:
        perm_list = read_permutation_dat_file("p6.dat")
        P = convert_all_permutations(perm_list)
    elif n == 7:
        perm_list = read_permutation_dat_file("p7.dat")
        P = convert_all_permutations(perm_list)

    return {
        "msize": n,
        "scale": scale,
        "J": np.arange(n),
        "A": A,
        "P": P,
        "I": np.arange(math.factorial(n)),
    }

def load_and_process_all(json_data):
    result = {}
    for key, entry in json_data.items():
        result[key] = process_entry(entry)
    return result