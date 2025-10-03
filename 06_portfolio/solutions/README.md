
# Solutions

The `solutions/` directory contains the results of solving BQP and UQO problems with various parameters. 
The solution files are organized based on the number of assets, allocation period, position limits, and other constraints encoded in their filenames:

- **Naming Convention:**
  - `aXXX`: Number of assets (e.g., `a200` for 200 assets).
  - `tXX`: Number of allocation days (e.g., `t10` for 10 days).
  - `sXX`: The seed for generation, can also be "orig" (e.g., `s01`).
  - `qX`: Weight of risk in the objective function (e.g., `q0.001`).

Higher `q` values indicate a greater emphasis on risk control within the model.

Some of the bigger instances ran out of memory when using Gurobi with 64 GB of RAM.
For profit values, look at hte [eval directory](./bqp/eval/).