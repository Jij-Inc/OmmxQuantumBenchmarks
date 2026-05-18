"""
Steiner Tree Packing Problem - Arc-based Optimized Implementation
Memory-efficient implementation using arc-based variable indexing
"""

import jijmodeling as jm


def create_steiner_tree_packing_model() -> jm.Problem:
    """Create Steiner Tree Packing optimization model using arc-based variables."""
    problem = jm.Problem("SteinerTreePackingArcBased", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        nets = problem.Natural("L", ndim=1, description="Nets")
        nodes = problem.Natural("V", ndim=1, description="Nodes")
        roots = problem.Natural("R", ndim=1, description="Root nodes")
        arcs = problem.Natural("A", ndim=2, description="Arcs")
        terminals = problem.Natural("T", ndim=1, description="Terminal nodes")
        normals = problem.Natural("N", ndim=1, description="Normal nodes")
        nodes_without_roots = problem.Natural(
            "VNR", ndim=1, description="Vertices excluding root nodes"
        )

        terminal_innet = problem.Natural(
            "innetT",
            ndim=2,
            description="Net assignments for terminals",
        )
        root_innet = problem.Natural(
            "innetR",
            ndim=2,
            description="Net assignments for roots",
        )
        arc_costs = problem.Float("arcCosts", ndim=1, description="Cost for each arc")
        net_cardinality = problem.Natural(
            "netCardinality",
            ndim=2,
            description="Cardinality of nets",
        )

        nA = arcs.len_at(0)
        nV = nodes.len_at(0)
        nL = nets.len_at(0)
        nR = roots.len_at(0)
        nT = terminals.len_at(0)
        nN = normals.len_at(0)
        nVNR = nodes_without_roots.len_at(0)

        big_m = nV * nV

        x = problem.BinaryVar(
            "x",
            shape=(nA, nT),
            description="x[a, t] = 1 if arc a carries flow for terminal t",
        )
        y = problem.BinaryVar(
            "y",
            shape=(nA, nL),
            description="y[a, l] = 1 if arc a is used by net l",
        )
        z = problem.BinaryVar(
            "z",
            shape=(nR, nT),
            description="z[r, t] = 1 if innetR[r] == innetT[t]",
        )

        # Objective: minimize sum arc_costs[a] * y[a, nets[l]]
        problem += jm.sum(arc_costs[a] * y[a, nets[l]] for a in nA for l in nL)

        # 1. ROOT FLOW OUT (Big-M)
        problem += problem.Constraint(
            "root_flow_out_big_m_lower",
            lambda t, r: jm.sum(
                x[a, t] for a in nA if arcs[a, 0] == roots[r]
            )
            >= z[r, t],
            domain=jm.product(nT, nR),
        )

        problem += problem.Constraint(
            "root_flow_out_big_m_upper",
            lambda t, r: jm.sum(x[a, t] for a in nA if arcs[a, 0] == roots[r])
            <= z[r, t] + big_m * (1 - z[r, t]),
            domain=jm.product(nT, nR),
        )

        problem += problem.Constraint(
            "root_flow_out_z_condition",
            lambda t, r: z[r, t]
            <= 1 - jm.abs(terminal_innet[t, 1] - root_innet[r, 1]) / big_m,
            domain=jm.product(nT, nR),
        )

        # 2. ROOT FLOW IN
        problem += problem.Constraint(
            "root_flow_in",
            lambda t, r: jm.sum(
                x[a, t] for a in nA if arcs[a, 1] == roots[r]
            )
            == 0,
            domain=jm.product(nT, nR),
        )

        # 3. TERMINAL FLOW OUT
        problem += problem.Constraint(
            "terms_flow_out",
            lambda t: jm.sum(
                x[a, t] for a in nA if arcs[a, 0] == terminals[t]
            )
            == 0,
            domain=nT,
        )

        # 4. TERMINAL FLOW IN
        problem += problem.Constraint(
            "terms_flow_in",
            lambda t: jm.sum(
                x[a, t] for a in nA if arcs[a, 1] == terminals[t]
            )
            == 1,
            domain=nT,
        )

        # 5. TERMINAL FLOW BALANCE SAME NET
        problem += problem.Constraint(
            "terms_flow_bal_same",
            lambda t, s: (
                jm.sum(x[a, t] for a in nA if arcs[a, 1] == terminals[s])
                - jm.sum(x[a, t] for a in nA if arcs[a, 0] == terminals[s])
                == 0
            ),
            domain=jm.product(nT, nT).filter(
                lambda t, s: (terminals[s] != terminals[t])
                & (terminal_innet[s, 1] == terminal_innet[t, 1])
            ),
        )

        # 6. TERMINAL FLOW BALANCE DIFFERENT NET
        problem += problem.Constraint(
            "terms_flow_bal_diff",
            lambda t, s: (
                jm.sum(x[a, t] for a in nA if arcs[a, 1] == terminals[s])
                - jm.sum(x[a, t] for a in nA if arcs[a, 0] == terminals[s])
                == 0
            ),
            domain=jm.product(nT, nT).filter(
                lambda t, s: terminal_innet[s, 1] != terminal_innet[t, 1]
            ),
        )

        # 7. NORMAL NODES FLOW BALANCE
        problem += problem.Constraint(
            "normal_flow_bal",
            lambda t, nn: (
                jm.sum(x[a, t] for a in nA if arcs[a, 0] == normals[nn])
                - jm.sum(x[a, t] for a in nA if arcs[a, 1] == normals[nn])
                == 0
            ),
            domain=jm.product(nT, nN),
        )

        # 8. BIND X TO Y
        problem += problem.Constraint(
            "bind_x_y",
            lambda a, l: jm.sum(
                x[a, t] for t in nT if terminal_innet[t, 1] == nets[l]
            )
            <= net_cardinality[l, 1] * y[a, nets[l]],
            domain=jm.product(nA, nL),
        )

        # 9. NODE DISJOINTNESS NON-ROOT
        problem += problem.Constraint(
            "disjoint_nonroot",
            lambda nwr: jm.sum(
                y[a, nets[l]]
                for a in nA
                for l in nL
                if arcs[a, 1] == nodes_without_roots[nwr]
            )
            <= 1,
            domain=nVNR,
        )

        # 10. ROOT NODE DISJOINTNESS
        problem += problem.Constraint(
            "disjoint_root",
            lambda r: jm.sum(
                y[a, nets[l]]
                for a in nA
                for l in nL
                if arcs[a, 1] == roots[r]
            )
            <= 0,
            domain=nR,
        )

    return problem
