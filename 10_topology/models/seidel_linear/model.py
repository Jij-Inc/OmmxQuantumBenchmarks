import jijmodeling as jm


def create_topology_model() -> jm.Problem:
    """Create Topology optimization model using Seidel linear formulation.

    Returns:
        JijModeling problem instance with all constraints and variables
        defined for the topology problem using seidel linear approach.
    """
    # Define data placeholders, which are from the data file.
    n = jm.Placeholder("nodes", description="Number of nodes in the graph")
    d = jm.Placeholder("degree", description="Maximum degree constraint")

    # Define bounds for diameter
    min_d = jm.Placeholder("minDiameter", description="Minimum diameter bound")
    max_d = jm.Placeholder("maxDiameter", description="Maximum diameter bound")

    # Define element objects
    s = jm.Element("s", belong_to=(0, n))
    t = jm.Element("t", belong_to=(0, n))
    i = jm.Element("i", belong_to=(0, n))
    k = jm.Element("k", belong_to=(0, max_d))
    j = jm.Element("j", belong_to=(0, n))

    # Define decision variables
    diameter = jm.IntegerVar(
        "diameter",
        lower_bound=min_d,
        upper_bound=max_d,
        description="Diameter of the graph",
    )

    # dist[s,t,k] - binary variable: 1 if there is a shortest path of length k between nodes s,t
    # Only defined for s < t (set F) and k in {0..max_d-1} (set D)
    dist = jm.BinaryVar(
        "dist",
        shape=(n, n, max_d),
        description="Distance variables: 1 if shortest path of length k between nodes s,t",
    )

    # y[s,t,i,k] - linearization variables for products of dist variables
    # Only defined for s < t (set F), i in N\{s,t}, and k in {0..max_d-2}
    y = jm.BinaryVar(
        "y",
        shape=(n, n, n, max_d),
        description="Linearization variables for distance products",
    )

    # Create problem with minimize objective
    problem = jm.Problem("TopologyOptimization", sense=jm.ProblemSense.MINIMIZE)

    # Objective: minimize diameter
    problem += diameter

    # Constraint 1: diameter constraint
    # forall <s,t> in F : 1 + sum <k> in D : (1 - dist[s,t,k]) <= diameter
    problem += jm.Constraint(
        "diameter_constraint",
        1 + jm.sum([k], 1 - dist[s, t, k]) <= diameter,
        forall=[s, (t, s < t)],
    )

    # Constraint 2: DistCalc - distance calculation constraint
    # forall <k> in D without {max_d-1} : forall <s,t> in F :
    # dist[s,t,k+1] <= dist[s,t,k] + sum <i> in N without {s,t}: y[s,t,i,k]
    problem += jm.Constraint(
        "DistCalc",
        dist[s, t, k + 1]
        <= dist[s, t, k] + jm.sum([(i, (i != s) & (i != t))], y[s, t, i, k]),
        forall=[s, (t, s < t), (k, k != max_d - 1)],
    )

    # Constraint 3: DistLinearize - linearization constraints
    # forall <k> in D without {max_d-1} : forall <s,t> in F : forall <i> in N without {s,t} :
    # y[s,t,i,k] <= dist[min(s,i),max(s,i),k] and y[s,t,i,k] <= dist[min(i,t),max(i,t),0]
    # First part: y[s,t,i,k] <= dist[min(s,i),max(s,i),k]
    problem += jm.Constraint(
        "DistLinearize_si",
        y[s, t, i, k] <= dist[jm.min(s, i), jm.max(s, i), k],
        forall=[s, (t, s < t), (i, (i != s) & (i != t)), (k, k != max_d - 1)],
    )
    # Second part: y[s,t,i,k] <= dist[min(i,t),max(i,t),0]
    problem += jm.Constraint(
        "DistLinearize_it",
        y[s, t, i, k] <= dist[jm.min(i, t), jm.max(i, t), 0],
        forall=[s, (t, s < t), (i, (i != s) & (i != t)), (k, k != max_d - 1)],
    )

    # Constraint 4: degreeButLast - degree constraint for all nodes except the last
    # forall <j> in N without {n-1} : sum <i> in N without {j}: dist[min(j,i),max(j,i),0] == d
    problem += jm.Constraint(
        "degreeButLast",
        jm.sum([(i, i != j)], dist[jm.min(j, i), jm.max(j, i), 0]) == d,
        forall=[(j, j != n - 1)],
    )

    # Constraint 5: degreeLast - degree constraint for the last node
    # sum <i> in N without {n-1} : dist[i,n-1,0] == if ((n * d) mod 2 == 0) then d else d-1 end
    # This constraint can be simplified to: d - (n*d mod 2)
    problem += jm.Constraint(
        "degreeLast",
        jm.sum([(i, i != n - 1)], dist[i, n - 1, 0]) == d - ((n * d) % 2),
        forall=[],
    )

    return problem
