import jijmodeling as jm


def create_topology_model() -> jm.Problem:
    """Create Topology optimization model.

    Returns:
        JijModeling problem instance with all constraints and variables
        defined for the topology problem.
    """
    # Define data placeholders, which are from the data file.
    n = jm.Placeholder("nodes", description="Number of nodes in the graph")
    d = jm.Placeholder("degree", description="Maximum degree constraint")

    # Define decision variables
    diameter = jm.IntegerVar(
        "diameter",
        lower_bound=0,
        upper_bound=n - 1,
        description="Diameter of the graph",
    )
    # SP[s,t] for s < t pairs - shortest path length between nodes s and t
    shortest_path = jm.IntegerVar(
        "SP",
        lower_bound=0,
        upper_bound=n - 1,
        shape=(n, n),
        description="Shortest path length between node pairs",
    )
    # z[i,j] for i < j - binary variable indicating if there's an edge between nodes i and j
    z = jm.BinaryVar("z", shape=(n, n), description="Edge existence between nodes")
    # x[s,t,i,j] - flow variables for shortest path from s to t using edge (i,j)
    x = jm.BinaryVar(
        "x", shape=(n, n, n, n), description="Flow variables for shortest paths"
    )

    # Define element objects
    s = jm.Element("s", belong_to=(0, n))
    t = jm.Element("t", belong_to=(0, n))
    i = jm.Element("i", belong_to=(0, n))
    j = jm.Element("j", belong_to=(0, n))

    # Create problem with minimize objective
    problem = jm.Problem("TopologyOptimization", sense=jm.ProblemSense.MINIMIZE)

    # Objective: minimize diameter
    problem += diameter

    # Constraint 1: diameter constraint - SP[s,t] <= diameter for all s < t
    problem += jm.Constraint(
        "diameter", shortest_path[s, t] <= diameter, forall=[s, (t, s < t)]
    )

    # Constraint 2: APSP - SP[s,t] == sum of flow variables for path from s to t
    problem += jm.Constraint(
        "APSP",
        shortest_path[s, t] == jm.sum([i, (j, i != j)], x[s, t, i, j]),
        forall=[s, (t, s < t)],
    )

    # Constraint 3: SPtransit - flow conservation at intermediate nodes
    problem += jm.Constraint(
        "SPtransit",
        jm.sum([(j, i != j)], x[s, t, i, j]) - jm.sum([(j, j != i)], x[s, t, j, i])
        == 0,
        forall=[s, (t, s < t), (i, (i != s) & (i != t))],
    )

    # Constraint 4: SPsource - flow out from source node
    problem += jm.Constraint(
        "SPsource",
        jm.sum([(j, s != j)], x[s, t, s, j]) - jm.sum([(j, j != s)], x[s, t, j, s])
        == 1,
        forall=[s, (t, s < t)],
    )

    # Constraint 5: SPtarget - flow into target node
    problem += jm.Constraint(
        "SPtarget",
        jm.sum([(j, t != j)], x[s, t, t, j]) - jm.sum([(j, j != t)], x[s, t, j, t])
        == -1,
        forall=[s, (t, s < t)],
    )

    # Constraint 6: degree constraint - each node has at most 'degree' edges
    problem += jm.Constraint(
        "degree_constraint",
        jm.sum([(j, i < j)], z[i, j]) + jm.sum([(j, j < i)], z[j, i]) <= d,
        forall=[i],
    )

    # Constraint 7: ZXlink - edge existence constraint linking z and x variables
    # z[i,j] >= x[s,t,i,j] and z[i,j] >= x[s,t,j,i] for i < j
    problem += jm.Constraint(
        "ZXlink_forward",
        z[i, j] >= x[s, t, i, j],
        forall=[s, (t, s < t), i, (j, i < j)],
    )

    problem += jm.Constraint(
        "ZXlink_backward",
        z[i, j] >= x[s, t, j, i],
        forall=[s, (t, s < t), i, (j, i < j)],
    )

    return problem
