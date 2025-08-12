def solve_c(constraint, known_vals, target_var_id):
    """
    Given a constraint whose function contains only linear, quadratic, and constant terms,
    and where only target_var_id is unknown, substitute the known values and solve for it.

    Parameters
    ----------
    constraint : ommx.v1.Constraint
        The constraint object whose .function defines the equation.
    known_vals : dict[int, float]
        A mapping from variable ID to its known value (0 or 1), excluding target_var_id.
    target_var_id : int
        The ID of the variable to solve for, e.g., 0.

    Returns
    -------
    float
        The value of the target variable that makes the constraint equal zero
        (assuming the coefficient for target_var_id is nonzero).
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
