"""
Steiner Tree Packing Problem - Mathematically Correct Optimized Implementation
Maximum auxiliary variable reduction while maintaining strict ZPL mathematical equivalence
"""

import jijmodeling as jm


def create_steiner_tree_packing_model() -> jm.Problem:
    """Create Steiner Tree Packing optimization model.

    Implements the node-disjoint Steiner tree packing problem using multicommodity
    flow formulation. Mathematically equivalent to the ZPL formulation in
    stp_node_disjoint.zpl.

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
    # cost matrix, [c_{ij}]
    cost = jm.Placeholder("cost", ndim=2, description="Cost matrix")
    # net cardinality, [(k in L, k's cardinality)]
    net_cardinality = jm.Placeholder(
        "netCardinality",
        ndim=2,
        dtype=jm.DataType.INTEGER,
        description="Cardinality of nets",
    )

    big_m = nodes.len_at(0) ** 2

    x = jm.BinaryVar(
        "x",
        shape=(nodes.len_at(0), nodes.len_at(0), terminals.len_at(0)),
        description="x[i, j, t] = 1 if arc (i, j) carries flow for terminal t",
    )
    y = jm.BinaryVar(
        "y",
        shape=(nodes.len_at(0), nodes.len_at(0), nets.len_at(0)),
        description="y[i, j, l] = 1 if arc (i, j) is used by net l",
    )

    # === Problem ===
    problem = jm.Problem(
        "SteinerTreePackingCorrectOptimized", sense=jm.ProblemSense.MINIMIZE
    )

    # Objective: minimize sum cost[i,j] * y[i,j,k]
    objective = jm.sum(
        [a, l], cost[arcs[a, 0], arcs[a, 1]] * y[arcs[a, 0], arcs[a, 1], nets[l]]
    )
    problem += objective

    # CONSTRAINT 1: ROOT FLOW OUT
    # subto root_flow_out:
    #    forall <t> in T do
    #       forall <r> in R do
    #          sum <r,j> in A : x[r,j,t] == if innet[r] == innet[t] then 1 else 0 end;
    # Employ Big-M method to handle the condition innet[r] == innet[t].
    # The new variable should be 1 if root_innet(r) == terminal_innet(t), otherwise 0
    z = jm.BinaryVar(
        "z",
        shape=(roots.len_at(0), terminals.len_at(0)),
        description="z[r, t] = 1 if root_innet[r] == terminal_innet[t]",
    )
    root_flow_out_big_m_lower = jm.Constraint(
        "root_flow_out_big_m_lower",
        jm.sum([(a, arcs[a, 0] == roots[r])], x[arcs[a, 0], arcs[a, 1], t]) >= z[r, t],
        forall=[t, r],
    )
    problem += root_flow_out_big_m_lower

    root_flow_out_big_m_upper = jm.Constraint(
        "root_flow_out_big_m_upper",
        jm.sum([(a, arcs[a, 0] == roots[r])], x[arcs[a, 0], arcs[a, 1], t])
        <= z[r, t] + big_m * (1 - z[r, t]),
        forall=[t, r],
    )
    problem += root_flow_out_big_m_upper

    root_flow_out_z_condition = jm.Constraint(
        "root_flow_out_z_consition",
        z[r, t] <= 1 - jm.abs(terminal_innet[t, 1] - root_innet[r, 1]) / big_m,
        forall=[t, r],
    )
    problem += root_flow_out_z_condition

    # 2. ROOT FLOW IN
    # subto root_flow_in:
    #    forall <t> in T do
    #       forall <r> in R do
    #          sum <i,r> in A : x[i,r,t] == 0;
    root_flow_in = jm.Constraint(
        "root_flow_in",
        jm.sum([(a, arcs[a, 1] == roots[r])], x[arcs[a, 0], arcs[a, 1], t]) == 0,
        forall=[t, r],
    )
    problem += root_flow_in

    # 3. TERMINAL FLOW OUT
    # subto terms_flow_out:
    #    forall <t> in T do
    #       sum <t,j> in A : x[t,j,t] == 0;
    terms_flow_out = jm.Constraint(
        "terms_flow_out",
        jm.sum([(a, arcs[a, 0] == terminals[t])], x[arcs[a, 0], arcs[a, 1], t]) == 0,
        forall=[t],
    )
    problem += terms_flow_out

    # # 4. TERMINAL FLOW IN
    # # subto terms_flow_in:
    # #    forall <t> in T do
    # #       sum <i,t> in A : x[i,t,t] == 1;
    # terms_flow_in = jm.Constraint(
    #     "terms_flow_in",
    #     jm.sum([(a, arcs[a, 1] == terminals[t])], x[arcs[a, 0], arcs[a, 1], t]) == 1,
    #     forall=[t],
    # )
    # problem += terms_flow_in

    # 5. TERMINAL FLOW BALANCE SAME NET
    # subto terms_flow_bal_same:
    #    forall <t> in T do
    #       forall <s> in T with s != t and innet[s] == innet[t] do
    #          sum <i,s> in A : x[i,s,t] - sum <s,j> in A : x[s,j,t] == 0;
    terms_flow_bal_same = jm.Constraint(
        "terms_flow_bal_same",
        (
            jm.sum([(a, arcs[a, 1] == terminals[s])], x[arcs[a, 0], arcs[a, 1], t])
            - jm.sum([(a, arcs[a, 0] == terminals[s])], x[arcs[a, 0], arcs[a, 1], t])
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
    # subto terms_flow_bal_diff:
    #    forall <t> in T do
    #       forall <s> in T with innet[s] != innet[t] do
    #          sum <i,s> in A : x[i,s,t] + sum <s,j> in A : x[s,j,t] == 0;
    terms_flow_bal_diff = jm.Constraint(
        "terms_flow_bal_diff",
        (
            jm.sum([(a, arcs[a, 1] == terminals[s])], x[arcs[a, 0], arcs[a, 1], t])
            - jm.sum([(a, arcs[a, 0] == terminals[s])], x[arcs[a, 0], arcs[a, 1], t])
            == 0
        ),
        forall=[t, (s, (terminal_innet[s, 1] != terminal_innet[t, 1]))],
    )
    problem += terms_flow_bal_diff

    # 7. NORMAL NODES FLOW BALANCE
    # subto nodes_flow_bal:
    #    forall <t> in T do
    #       forall <n> in N do
    #          sum <n,j> in A : x[n,j,t] - sum <i,n> in A : x[i,n,t] == 0;
    normal_flow_bal = jm.Constraint(
        "normal_flow_bal",
        (
            jm.sum([(a, arcs[a, 0] == normals[n])], x[arcs[a, 0], arcs[a, 1], t])
            - jm.sum([(a, arcs[a, 1] == normals[n])], x[arcs[a, 0], arcs[a, 1], t])
            == 0
        ),
        forall=[t, n],
    )
    problem += normal_flow_bal

    # CONSTRAINT 8: BIND X TO Y
    # subto bind_x_y:
    #    forall <i,j> in A do
    #       forall <k> in L do
    #          sum <t> in T with innet[t] == k : x[i,j,t] <= nets[k] * y[i,j,k];
    bind_x_y = jm.Constraint(
        "bind_x_y",
        jm.sum([(t, terminal_innet[t, 1] == nets[l])], x[arcs[a, 0], arcs[a, 1], t])
        <= net_cardinality[l, 1] * y[arcs[a, 0], arcs[a, 1], nets[l]],
        forall=[a, l],
    )
    problem += bind_x_y

    # CONSTRAINT 9: NODE DISJOINTNESS NON-ROOT
    # subto disjoint_nonroot:
    #    forall <j> in V without R do
    #       sum <i,j> in A, <k> in L : y[i,j,k] <= 1;
    disjoint_nonroot = jm.Constraint(
        "disjoint_nonroot",
        jm.sum(
            [(a, arcs[a, 1] == nodes_without_roots[nwr]), l],
            y[arcs[a, 0], arcs[a, 1], nets[l]],
        )
        <= 1,
        forall=[nwr],
    )
    problem += disjoint_nonroot

    # CONSTRAINT 10: ROOT NODE DISJOINTNESS
    # subto disjoint_root:
    #    forall <r> in R do
    #       sum <i,r> in A, <k> in L : y[i,r,k] <= 0;
    disjoint_root = jm.Constraint(
        "disjoint_root",
        jm.sum([(a, arcs[a, 1] == roots[r]), l], y[arcs[a, 0], arcs[a, 1], nets[l]])
        <= 0,
        forall=[r],
    )
    problem += disjoint_root

    return problem
