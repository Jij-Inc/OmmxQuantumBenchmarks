# batch_portfolio_by_q.py
import os
import glob
import jijmodeling as jm
from ommx.artifact import ArtifactBuilder
import re
from model import createPortfolioBqpModel
from dat_reader import build_portfolio_instance_for_createPortfolioBqpModel
from sol_reader import parse_solution_ordered_dict_debug_v7
from eval_portfolio import eval_portfolio_from_solution, extract_solution_arrays_from_df


# ------------------ Locate sol folders ------------------
def stem_from_instance_base(inst_base: str) -> str:
    """po_a010_t10_orig -> a010_t10_orig"""
    return inst_base[3:] if inst_base.startswith("po_") else inst_base


def collect_sol_files_for_instance(sol_root: str, inst_base: str):
    """Find the matching folder under solutions/bqp/bqp and collect all .sol files."""
    eval_root = os.path.join(sol_root, "bqp", "bqp")
    stem = stem_from_instance_base(inst_base)
    cand_dirs = [
        d for d in glob.glob(os.path.join(eval_root, f"{stem}*")) if os.path.isdir(d)
    ]
    sol_files = []
    for d in cand_dirs:
        sol_files.extend(sorted(glob.glob(os.path.join(d, "*.sol"))))
    return sol_files


# ------------------ Helper ------------------
def extract_q_from_fname(fname: str) -> float:
    # fname example: bqp_eval_a010_t10_q0.000001_b004.sol
    m = re.search(r"_q([0-9.]+)_b", fname)
    if m:
        return float(m.group(1))
    raise ValueError(f"cannot parse q from {fname}")


def read_symbol_to_i(stock_file: str, sort: bool = False) -> dict[str, int]:
    """
    Read stock list file and construct {symbol: index} mapping.

    Args:
        stock_file (str): Path to the stock prices file.
        sort (bool): If True, symbols will be sorted alphabetically.
                     If False, symbols keep first-appearance order.

    Returns:
        dict[str, int]: Mapping from symbol string to index.
    """
    symbols = []
    with open(stock_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sym = parts[1]
            if sym not in symbols:
                symbols.append(sym)

    if sort:
        symbols = sorted(symbols)

    symbol_to_i = {s: i for i, s in enumerate(symbols)}
    return symbol_to_i


# ------------------ Main routine ------------------
def batch_process(
    inst_dir: str = "../../instances",
    sol_root: str = "../../solutions",
    output_directory: str = "./ommx_output",
):
    os.makedirs(output_directory, exist_ok=True)
    problem = createPortfolioBqpModel()

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

            sol_files = collect_sol_files_for_instance(sol_root, base)
            if not sol_files:
                print(f"[{base}] no sol files found, skip")
                continue
            for sol_path in sol_files:
                try:
                    # Extract q from filename, e.g., "..._q0.01.opt.sol"
                    fname = os.path.basename(sol_path)
                    q_val = None
                    b_val = None

                    # q
                    m_q = re.search(r"_q([0-9.]+)_b", fname)
                    if m_q:
                        try:
                            q_val = float(m_q.group(1))
                        except ValueError:
                            q_val = None

                    # b
                    m_b = re.search(r"_b(\d+)", fname)
                    if m_b:
                        try:
                            b_val = int(m_b.group(1))
                        except ValueError:
                            b_val = None

                    instance = build_portfolio_instance_for_createPortfolioBqpModel(
                        cov_file=cov_file,
                        price_file=price_file,
                        q=q_val,
                        b_tot=b_val,
                    )
                    interpreter = jm.Interpreter(instance)
                    ommx_instance = interpreter.eval_problem(problem)

                    # Parse solution
                    nSc, nTx, nCs1, nCs2 = (
                        len(instance["setSc"]),
                        len(instance["setTx"]),
                        len(instance["setCs1"]),
                        len(instance["setCs2"]),
                    )
                    symbol_to_i = read_symbol_to_i(price_file, sort=True)
                    sol_dict, _, obj_from_sol = parse_solution_ordered_dict_debug_v7(
                        sol_path,
                        symbol_to_i=symbol_to_i,
                        nSc=nSc,
                        nTx=nTx,
                        nCs1=nCs1,
                        nCs2=nCs2,
                    )

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

                    # Evaluate the objective value
                    df = sol_eval.decision_variables_df
                    nS = len(instance["setS"])
                    nSc = len(instance["setSc"])
                    nSl = len(instance["setSl"])
                    nT = len(instance["setTx"])
                    nCs1 = len(instance["setCs1"])
                    nCs2 = len(instance["setCs2"])
                    x_sol, s1_sol, s2_sol = extract_solution_arrays_from_df(
                        df, nS, nSc, nSl, nT, nCs1, nCs2
                    )
                    params = instance
                    eval_result = eval_portfolio_from_solution(
                        xVal=x_sol,
                        s1Val=s1_sol,
                        s2Val=s2_sol,
                        up=params["up"],
                        cov=params["cov"],
                        slPnl=params["slPnl"],
                        slCash=params["slCash"],
                        slNeg=params["slNeg"],
                        pow2Cs1=params["pow2Cs1"],
                        pow2Cs2=params["pow2Cs2"],
                        isFirst=params["isFirst"],
                        isMid=params["isMid"],
                        isLast=params["isLast"],
                        nextT=params["nextT"],
                        prevT=params["prevT"],
                        unit=params["unit"],
                        delta=params["delta"],
                        rhoC=params["rhoC"],
                        rhoS=params["rhoS"],
                        q=params["q"],
                        upscale=params["upscale"],
                        bCsh=params["bCsh"],
                        bTot=params["bTot"],
                    )
                    obj_from_eval = eval_result["obj"]
                    feasible_c2 = eval_result["c2_ok"]
                    feasible_c3 = eval_result["c3_ok"]
                    print(
                        f"Obj from eval process={obj_from_eval}, c2 feasible={feasible_c2}, c3 feasible={feasible_c3}"
                    )
                    print(f"Obj from eval sol_file={obj_from_sol}")
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
