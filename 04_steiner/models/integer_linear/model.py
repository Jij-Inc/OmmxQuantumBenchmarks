"""
Steiner Tree Packing Problem - Arc-based Optimized Implementation
Memory-efficient implementation using arc-based variable indexing
"""

import jijmodeling as jm


def create_steiner_tree_packing_model() -> jm.Problem:
    """Create Steiner Tree Packing optimization model using arc-based variables.

    Implements the node-disjoint Steiner tree packing problem using multicommodity
    flow formulation with arc-based indexing for memory efficiency.

    Returns:
        JijModeling problem instance with all constraints and variables
        defined for the Steiner tree packing problem.
    """
    # nets, [0, 1, 2, ..., |L| - 1]
    nets = jm.Placeholder("L", ndim=1, dtype=jm.DataType.INTEGER, description="Nets")
    l = jm.Element("l", belong_to=(0, nets.len_at(0)))
    # nodes, [0, 1, 2, ..., |V| - 1]
    nodes = jm.Placeholder("V", ndim=1, dtype=jm.DataType.INTEGER, description="Nodes")
    v = jm.Element("v", belong_to=(0, nodes.len_at(0)))
    # root nodes, [r, ...]
    roots = jm.Placeholder(
        "R", ndim=1, dtype=jm.DataType.INTEGER, description="Root nodes"
    )
    r = jm.Element("r", belong_to=(0, roots.len_at(0)))
    # arcs (node to node), [(i, j), ...] subset (V * V)
    arcs = jm.Placeholder("A", ndim=2, dtype=jm.DataType.INTEGER, description="Arcs")
    a = jm.Element("a", belong_to=(0, arcs.len_at(0)))
    # terminal nodes, S - R = [t, ...]
    terminals = jm.Placeholder(
        "T", ndim=1, dtype=jm.DataType.INTEGER, description="Terminal nodes"
    )
    t = jm.Element("t", belong_to=(0, terminals.len_at(0)))
    s = jm.Element("s", belong_to=(0, terminals.len_at(0)))
    # normal nodes, V - S = [n, ...]
    normals = jm.Placeholder(
        "N", ndim=1, dtype=jm.DataType.INTEGER, description="Normal nodes"
    )
    n = jm.Element("n", belong_to=(0, normals.len_at(0)))
    # vertices excluding root nodes, V - R = [v, ...]
    nodes_without_roots = jm.Placeholder(
        "VNR",
        ndim=1,
        dtype=jm.DataType.INTEGER,
        description="Vertices excluding root nodes",
    )
    nwr = jm.Element("vnr", belong_to=(0, nodes_without_roots.len_at(0)))

    # terminal innet, [(terminal, k in L)]
    terminal_innet = jm.Placeholder(
        "innetT",
        ndim=2,
        dtype=jm.DataType.INTEGER,
        description="Net assignments for terminals",
    )
    # root innet, [(root, k in L)]
    root_innet = jm.Placeholder(
        "innetR",
        ndim=2,
        dtype=jm.DataType.INTEGER,
        description="Net assignments for roots",
    )
    # arc costs, [cost for each arc]
    arc_costs = jm.Placeholder("arcCosts", ndim=1, description="Cost for each arc")
    # net cardinality, [(k in L, k's cardinality)]
    net_cardinality = jm.Placeholder(
        "netCardinality",
        ndim=2,
        dtype=jm.DataType.INTEGER,
        description="Cardinality of nets",
    )

    big_m = nodes.len_at(0) ** 2

    # Arc-based variables - much more memory efficient
    x = jm.BinaryVar(
        "x",
        shape=(arcs.len_at(0), terminals.len_at(0)),
        description="x[a, t] = 1 if arc a carries flow for terminal t",
    )
    y = jm.BinaryVar(
        "y",
        shape=(arcs.len_at(0), nets.len_at(0)),
        description="y[a, l] = 1 if arc a is used by net l",
    )

    # === Problem ===
    problem = jm.Problem("SteinerTreePackingArcBased", sense=jm.ProblemSense.MINIMIZE)

    # Objective: minimize sum arc_costs[a] * y[a,k]
    objective = jm.sum([a, l], arc_costs[a] * y[a, nets[l]])
    problem += objective

    # CONSTRAINT 1: ROOT FLOW OUT
    # Employ Big-M method to handle the condition innet[r] == innet[t].
    # The new variable should be 1 if root_innet(r) == terminal_innet(t), otherwise 0
    z = jm.BinaryVar(
        "z",
        shape=(roots.len_at(0), terminals.len_at(0)),
        description="z[r, t] = 1 if innetR[r] == innetT[t]",
    )
    root_flow_out_big_m_lower = jm.Constraint(
        "root_flow_out_big_m_lower",
        jm.sum([(a, arcs[a, 0] == roots[r])], x[a, t]) >= z[r, t],
        forall=[t, r],
    )
    problem += root_flow_out_big_m_lower

    root_flow_out_big_m_upper = jm.Constraint(
        "root_flow_out_big_m_upper",
        jm.sum([(a, arcs[a, 0] == roots[r])], x[a, t])
        <= z[r, t] + big_m * (1 - z[r, t]),
        forall=[t, r],
    )
    problem += root_flow_out_big_m_upper

    root_flow_out_z_condition = jm.Constraint(
        "root_flow_out_z_condition",
        z[r, t] <= 1 - jm.abs(terminal_innet[t, 1] - root_innet[r, 1]) / big_m,
        forall=[t, r],
    )
    problem += root_flow_out_z_condition

    # 2. ROOT FLOW IN
    root_flow_in = jm.Constraint(
        "root_flow_in",
        jm.sum([(a, arcs[a, 1] == roots[r])], x[a, t]) == 0,
        forall=[t, r],
    )
    problem += root_flow_in

    # 3. TERMINAL FLOW OUT
    terms_flow_out = jm.Constraint(
        "terms_flow_out",
        jm.sum([(a, arcs[a, 0] == terminals[t])], x[a, t]) == 0,
        forall=[t],
    )
    problem += terms_flow_out

    # 4. TERMINAL FLOW IN
    terms_flow_in = jm.Constraint(
        "terms_flow_in",
        jm.sum([(a, arcs[a, 1] == terminals[t])], x[a, t]) == 1,
        forall=[t],
    )
    problem += terms_flow_in

    # 5. TERMINAL FLOW BALANCE SAME NET
    terms_flow_bal_same = jm.Constraint(
        "terms_flow_bal_same",
        (
            jm.sum([(a, arcs[a, 1] == terminals[s])], x[a, t])
            - jm.sum([(a, arcs[a, 0] == terminals[s])], x[a, t])
            == 0
        ),
        forall=[
            t,
            (
                s,
                (terminals[s] != terminals[t])
                & (terminal_innet[s, 1] == terminal_innet[t, 1]),
            ),
        ],
    )
    problem += terms_flow_bal_same

    # 6. TERMINAL FLOW BALANCE DIFFERENT NET
    terms_flow_bal_diff = jm.Constraint(
        "terms_flow_bal_diff",
        (
            jm.sum([(a, arcs[a, 1] == terminals[s])], x[a, t])
            - jm.sum([(a, arcs[a, 0] == terminals[s])], x[a, t])
            == 0
        ),
        forall=[t, (s, (terminal_innet[s, 1] != terminal_innet[t, 1]))],
    )
    problem += terms_flow_bal_diff

    # 7. NORMAL NODES FLOW BALANCE
    normal_flow_bal = jm.Constraint(
        "normal_flow_bal",
        (
            jm.sum([(a, arcs[a, 0] == normals[n])], x[a, t])
            - jm.sum([(a, arcs[a, 1] == normals[n])], x[a, t])
            == 0
        ),
        forall=[t, n],
    )
    problem += normal_flow_bal

    # CONSTRAINT 8: BIND X TO Y
    bind_x_y = jm.Constraint(
        "bind_x_y",
        jm.sum([(t, terminal_innet[t, 1] == nets[l])], x[a, t])
        <= net_cardinality[l, 1] * y[a, nets[l]],
        forall=[a, l],
    )
    problem += bind_x_y

    # CONSTRAINT 9: NODE DISJOINTNESS NON-ROOT
    disjoint_nonroot = jm.Constraint(
        "disjoint_nonroot",
        jm.sum(
            [(a, arcs[a, 1] == nodes_without_roots[nwr]), l],
            y[a, nets[l]],
        )
        <= 1,
        forall=[nwr],
    )
    problem += disjoint_nonroot

    # CONSTRAINT 10: ROOT NODE DISJOINTNESS
    disjoint_root = jm.Constraint(
        "disjoint_root",
        jm.sum([(a, arcs[a, 1] == roots[r]), l], y[a, nets[l]]) <= 0,
        forall=[r],
    )
    problem += disjoint_root

    return problem
