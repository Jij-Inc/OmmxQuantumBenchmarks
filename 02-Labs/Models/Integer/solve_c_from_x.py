import ommx
from typing import Dict


def solve_c(
    constraint: ommx.v1.Instance.constraints , known_vals: Dict[int, float], target_var_id: int
) -> float:
    """
    Solves for the value of a single unknown binary/continuous variable in a constraint
    when all other variables in the constraint are known.

    Args:
        constraint (ommx.v1.Instance.constraints): 
            A constraint object from an OMMX instance whose `.function` contains 
            linear, quadratic, and constant terms defining the equation.
        known_vals (Dict[int, float]): 
            A mapping from variable IDs (excluding the target) to their known values.
            Keys are variable IDs, values are typically 0.0 or 1.0 for binary variables,
            but can be continuous for non-binary problems.
        target_var_id (int): 
            The ID of the single unknown variable to solve for.

    Returns:
        float: 
            The computed value of the target variable that satisfies the constraint 
            equation, assuming its coefficient in the linear terms is non-zero.
    """
    fn = constraint.function

    def _get_value(obj, attr):
        value = getattr(obj, attr)
        return value() if callable(value) else value

    # Retrieve linear, quadratic, and constant components
    linear = _get_value(fn, "linear_terms")
    quadratic = _get_value(fn, "quadratic_terms")
    constant = _get_value(fn, "constant_term")

    # Sum contributions from known linear terms (skip the target variable)
    sum_lin = 0.0
    for vid, coeff in linear.items():
        if vid == target_var_id:
            continue
        sum_lin += coeff * known_vals.get(vid, 0.0)

    # Sum contributions from known quadratic terms (ignore any term involving the target)
    sum_quad = 0.0
    for (i, j), coeff in quadratic.items():
        sum_quad += coeff * known_vals.get(i, 0.0) * known_vals.get(j, 0.0)

    # The constant term of the function
    const = constant

    # Coefficient of the target variable in the linear terms
    a = linear[target_var_id]

    # Solve for x in: a * x + (sum_lin + sum_quad + const) == 0
    x = -(const + sum_lin + sum_quad) / a
    return x
