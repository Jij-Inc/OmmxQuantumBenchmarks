# batch_portfolio_by_q.py
import os
import glob
import jijmodeling as jm
from ommx.artifact import ArtifactBuilder
import re
from model import createPortfolioUQOModel
from dat_reader import build_portfolio_instance_for_createPortfolioUqoModel
from sol_reader import parse_x_hash_kv


# ------------------ Locate sol folders ------------------
def stem_from_instance_base(inst_base: str) -> str:
    """po_a010_t10_orig -> a010_t10_orig"""
    return inst_base[3:] if inst_base.startswith("po_") else inst_base


def collect_mst_files_for_instance(sol_root: str, inst_base: str):
    """Find the matching folder under solutions/uqo and collect all .sol.mst files."""
    eval_root = os.path.join(sol_root, "uqo")
    stem = stem_from_instance_base(inst_base)
    cand_dirs = [
        d for d in glob.glob(os.path.join(eval_root, f"{stem}*")) if os.path.isdir(d)
    ]
    sol_files = []
    for d in cand_dirs:
        sol_files.extend(sorted(glob.glob(os.path.join(d, "*.mst"))))
    return sol_files


# ------------------ Helper ------------------
def extract_q_from_fname(fname: str) -> float:
    # fname example: bqp_eval_a010_t10_q0.000001_b004.sol
    m = re.search(r"_q([0-9.]+)_b", fname)
    if m:
        return float(m.group(1))
    raise ValueError(f"cannot parse q from {fname}")


def extract_q_b_from_fname(path: str) -> tuple[float, int]:
    """Extract q and b from the filename. Supports patterns like
    '..._q0_b050.sol.mst'.

    Args:
        path: Full path or filename.

    Returns:
        (q, b) where q may be None (no match) and b may be None (no match).
    """
    name = path.split("/")[-1]
    # q: allow integer/decimal/scientific notation; b: integer (leading zeros allowed)
    m = re.search(
        r"_q([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)_b(\d+)\.sol(?:\.mst)?$",
        name,
        flags=re.IGNORECASE,
    )
    if not m:
        return None, None
    q = float(m.group(1))
    b = int(m.group(2))
    return q, b


# ------------------ Main routine ------------------
def batch_process(
    inst_dir: str = "../../instances",
    sol_root: str = "../../solutions",
    output_directory: str = "./ommx_output",
):
    os.makedirs(output_directory, exist_ok=True)
    problem = createPortfolioUQOModel()

    inst_folders = sorted(
        d for d in glob.glob(os.path.join(inst_dir, "*")) if os.path.isdir(d)
    )
    processed, errors = 0, 0

    for inst_path in inst_folders:
        base = os.path.basename(inst_path)
        try:
            cov_file = os.path.join(inst_path, "covariance_matrices.txt")
            price_file = os.path.join(inst_path, "stock_prices.txt")
            if not (os.path.exists(cov_file) and os.path.exists(price_file)):
                print(f"[{base}] missing cov/price, skip")
                continue

            sol_files = collect_mst_files_for_instance(sol_root, base)
            if not sol_files:
                print(f"[{base}] no sol files found, skip")
                continue
            for sol_path in sol_files:
                try:
                    # Extract q from filename, e.g., "..._q0.01.opt.sol"
                    fname = os.path.basename(sol_path)
                    q_val = None
                    b_val = None

                    q_val, b_val = extract_q_b_from_fname(sol_path)

                    instance = build_portfolio_instance_for_createPortfolioUqoModel(
                        cov_file=cov_file,
                        price_file=price_file,
                        q=q_val,
                        b_tot=b_val,
                    )
                    interpreter = jm.Interpreter(instance)
                    ommx_instance = interpreter.eval_problem(problem)

                    # Parse solution
                    sol_dict = parse_x_hash_kv(sol_path)
                    sol_eval = ommx_instance.evaluate(sol_dict)

                    # Create for the dir
                    base_out_dir = os.path.join(output_directory, base)
                    os.makedirs(base_out_dir, exist_ok=True)
                    out_path = os.path.join(base_out_dir, f"{fname}.ommx")
                    if os.path.exists(out_path):
                        os.remove(out_path)

                    builder = ArtifactBuilder.new_archive_unnamed(out_path)
                    builder.add_instance(ommx_instance)
                    if sol_eval is not None:
                        builder.add_solution(sol_eval)
                    builder.build()
                    print(f"  ✓ saved {out_path}\n")
                    processed += 1
                    print(
                        f"[{base}] {fname}: q={q_val}, b={b_val}, obj={sol_eval.objective}, feasible={sol_eval.feasible}"
                    )

                except Exception as sol_err:
                    print(
                        f"[{base}] {os.path.basename(sol_path)} eval failed: {sol_err}"
                    )

        except Exception as e:
            print(f"[{base}] ERROR: {e}\n")
            errors += 1

    print(f"Batch done — processed={processed}, errors={errors}")


if __name__ == "__main__":
    batch_process(
        inst_dir="../../instances",
        sol_root="../../solutions",
        output_directory="./ommx_output",
    )
