import jijmodeling as jm
import ommx.v1
import QOBLIBReader

# Placeholders for data from file
I_set = jm.Placeholder("I", ndim=1, description="Set I")
J_set = jm.Placeholder("J", ndim=1, description="Set J")
a = jm.Placeholder("a", ndim=2, description="Parameter a")
b = jm.Placeholder("b", ndim=1, description="Parameter b")

# Decision variables
# Note: J\{0} is handled within the constraints.  We define x over all of J.
x = jm.BinaryVar("x", shape=J_set.shape, description="Variable x")

# Elements for indexing
i = jm.Element("i", belong_to=I_set)
j = jm.Element("j", belong_to=J_set)

# Problem definition
problem = jm.Problem("Marketsplit_unconstrained", sense=jm.ProblemSense.MINIMIZE)

# Objective function
problem += jm.sum(
    [i], (b[i] - jm.sum([j], a[i, j] * x[j]))**2
)

# need the dat from QOBLB in /instances
instance_data = QOBLIBReader("ms_03_050_002.dat").read_dat_file()
ommx.v1.Instance = jm.Interpreter(instance_data).eval_problem(problem)