"""This script uploads OMMX data to a GitHub repository using Minto."""

import argparse
import glob
import os
from pathlib import Path
import time

import minto
import ommx.v1
from ommx.artifact import Artifact


def get_ommx_files(dir_path: str) -> list[str]:
    """Get all OMMX files from the specified directory.

    Args:
        dir_path (str): the target directory path.

    Returns:
        list[str]: ommx file paths found in the directory.
    """

    return glob.glob(os.path.join(dir_path, "*.ommx"))


def load_artifact(filepath: str) -> Artifact:
    """Load an OMMX artifact from the specified file path.

    Args:
        filepath (str): the target ommx file path.

    Returns:
        Artifact: the loaded OMMX artifact.
    """
    return Artifact.load_archive(filepath)


def create_minto_experiment(name: str, dir_path: str) -> minto.Experiment | None:
    """Create a Minto experiment that contains all ommx data with the given title.

    Args:
        name (str): the name of the experiment.
        dir_path (str): the directory path where the OMMX files are located.

    Returns:
        minto.Experiment | None: a Minto experiment instance containing the OMMX data.
    """
    experiment = minto.Experiment(name=name, auto_saving=False)
    ommx_filepaths = get_ommx_files(dir_path)
    if len(ommx_filepaths) == 0:
        return None

    for filepath in ommx_filepaths:
        artifact = load_artifact(filepath)
        filename = Path(filepath).stem
        if artifact.instance is not None:
            experiment.log_instance(instance_name=filename, instance=artifact.instance)
        else:
            raise ValueError(f"Instance is None for file: {filepath}")
        if artifact.solution is not None:
            experiment.log_solution(solution_name=filename, solution=artifact.solution)
        else:
            print(f"Warning: Solution is None for file: {filepath}")

    return experiment


def verify_minto_experiment(experiment: minto.Experiment, dir_path: str) -> bool:
    """Verify the Minto experiment by checking if it contains all OMMX data.

    Args:
        experiment (minto.Experiment): a Minto experiment to be verified.
        dir_path (str): a directory path where the OMMX files are located.

    Returns:
        bool: True if the experiment contains all OMMX data, False otherwise.
    """
    # Return False if the experiment does not have instances.
    # See https://github.com/Jij-Inc/minto/blob/b926c506626cb2a8fdf44a81acf5a16621886a82/minto/v1/datastore.py#L278
    if experiment.get_current_datastore().instances == {}:
        print("No instances found in the experiment.")
        return False

    ommx_filepaths = get_ommx_files(dir_path)
    for filepath in ommx_filepaths:
        loaded_artifact = load_artifact(filepath)

        # Check if the specified ommx file has an instance.
        loaded_instance = loaded_artifact.instance
        if loaded_instance is None:
            print(f"Warning: Instance is None for file: {filepath}")
            continue

        # Get the instance name and the corresponding instance from the experiment.
        instance_name = Path(filepath).stem
        experiment_instance = experiment.get_current_datastore().instances.get(
            instance_name, None
        )

        # Return False if the instance is not found in the experiment.
        if experiment_instance is None:
            print(f"Instance {instance_name} is missing in the experiment.")
            return False
        # Return False if the instance is not
        if not are_same_instances(experiment_instance, loaded_instance):
            print(
                f"Instance {instance_name} in the given experiment is not the same as the loaded instance."
            )
            return False

        # Check if the specified ommx file has a solution.
        loaded_solution = loaded_artifact.solution
        if loaded_solution is None:
            print(f"Warning: Solution is None for file: {filepath}")
            continue

        # Get the corresponding solution from the experiment.
        experiment_solution = experiment.get_current_datastore().solutions.get(
            instance_name, None
        )
        # Return False if the solution is not found in the experiment.
        if experiment_solution is None:
            print(f"Solution {instance_name} is missing in the experiment.")
            return False
        # Return False if the solution is not the same as the loaded solution.
        if not are_same_solutions(experiment_solution, loaded_solution):
            print(
                f"Solution {instance_name} in the given experiment is not the same as the loaded solution."
            )
            return False

    return True


def are_same_instances(
    instance1: ommx.v1.Instance, instance2: ommx.v1.Instance
) -> bool:
    """Check if two ommx instances are the same.

    Args:
        instance1 (ommx.v1.Instance): the first ommx instance.
        instance2 (ommx.v1.Instance): the second ommx instance.

    Returns:
        bool: True if the instances are the same, False otherwise.
    """
    # Check the decision variables of the instances.
    instance1_vars = instance1.decision_variables
    instance2_vars = instance2.decision_variables
    # Return False if the number of decision variables is different.
    if len(instance1_vars) != len(instance2_vars):
        print("The number of decision variables is different.")
        return False
    # Return False if the IDs, kinds, or bounds of the decision variables are different.
    for var1, var2 in zip(instance1_vars, instance2_vars):
        if var1.id != var2.id or var1.kind != var2.kind or var1.bound != var2.bound:
            print(
                f"The decision variables are different. ",
                f"(ID: {var1.id} for the first instance and ID: {var2.id} for the second instance) ",
                f"The first instance variable kind: {var1.kind}, bound: {var1.bound}, ",
                f"The second instance variable kind: {var2.kind}, bound: {var2.bound}. ",
            )
            return False

    # Check the constraints of the instances.
    instance1_constraints = instance1.constraints
    instance2_constraints = instance2.constraints
    # Return False if the number of constraints is different.
    if len(instance1_constraints) != len(instance2_constraints):
        print("The number of constraints is different.")
        return False
    # Return False if the IDs or equality of the constraints are different.
    for c1, c2 in zip(instance1_constraints, instance2_constraints):
        if c1.id != c2.id or c1.equality != c2.equality:
            print(
                f"The constraints are different. ",
                f"(ID: {c1.id} for the first instance and ID: {c2.id} for the second instance) ",
                f"The first instance constraint equality: {c1.equality}, ",
                f"The second instance constraint equality: {c2.equality}. ",
            )
            return False

    # Check the objective functions of the instances.
    if not instance1.objective.almost_equal(instance2.objective):
        print("The objective functions are different.")
        return False

    return True


def are_same_solutions(
    solution1: ommx.v1.Solution, solution2: ommx.v1.Solution
) -> bool:
    # Return False if the objective values are different.
    if solution1.objective != solution2.objective:
        print(
            f"The objective values of the solutions are different: {solution1.objective} vs {solution2.objective}"
        )
        return False
    # Return False if the decision variables are different.
    if solution1.state.entries != solution2.state.entries:
        print("The state entries of the solutions are different.")
        for var_id in set(solution1.state.entries.keys()) | set(
            solution2.state.entries.keys()
        ):
            val1 = solution1.state.entries.get(var_id, "Undefined")
            val2 = solution2.state.entries.get(var_id, "Undefined")
            if val1 != val2:
                print(f"Variable ID {var_id}: {val1} vs {val2}")
        return False
    # Return False if the feasibility of the solutions is different.
    if solution1.feasible != solution2.feasible:
        print(
            f"The feasibility of the solutions is different: {solution1.feasible} vs {solution2.feasible}"
        )
        return False

    return True


def upload_ommx(
    model_dir_path: str,
    dataset_name: str,
    github_username: str,
    github_pat: str,
    org: str = "Jij-Inc",
    repo: str = "OMMX-OBLIB",
) -> None:
    """Upload Minto experiments having OMMX data to GitHub repository.

    Args:
        model_dir_path (str): the path to the directory containing OMMX models.
        dataset_name (str): a name of the target dataset.
        github_username (str): a GitHub username.
        github_pat (str): a GitHub personal access token (PAT).
        org (str, optional): the GitHub organization name. Defaults to "Jij-Inc".
        repo (str, optional): the GitHub repository name. Defaults to "OMMX-OBLIB".
    """
    print("===========================")
    print(f"Model directory path: {model_dir_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"GitHub Organisation: {org}")
    print(f"GitHub Repository: {repo}")
    print("===========================")

    # Set up the github authentication environment variables.
    os.environ["OMMX_BASIC_AUTH_DOMAIN"] = "ghcr.io"
    os.environ["OMMX_BASIC_AUTH_USERNAME"] = github_username
    os.environ["OMMX_BASIC_AUTH_PASSWORD"] = github_pat

    # Obtain all directories in the target model directory.
    items = os.listdir(model_dir_path)
    dir_names = []
    for item in items:
        item_path = os.path.join(model_dir_path, item)
        if os.path.isdir(item_path):
            dir_names.append(item)

    for dir_name in dir_names:
        # Create a Minto experiment for each directory.
        name = f"{dataset_name}/{dir_name}"
        dir_path = os.path.join(model_dir_path, dir_name, "ommx_output")
        experiment = create_minto_experiment(name=name, dir_path=dir_path)
        if experiment is None:
            print(f"No OMMX data found in {dir_path}. Skipping this directory.")
            continue

        # Validate the Minto experiment.
        is_valid = verify_minto_experiment(experiment=experiment, dir_path=dir_path)

        if not is_valid:
            print(
                f"The experiment {name} is not valid. We won't push it to GitHub for now."
            )
        else:
            # Push the experiment to GitHub.
            print(f"Uploading {name} to GitHub...")
            start_time = time.time()
            artifact = experiment.push_github(org=org, repo=repo, name=name)
            end_time = time.time()
            print(f"Uploaded {name} to GitHub: {artifact.image_name}.")
            print(f"Upload time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload OMMX data to GitHub.")
    parser.add_argument(
        "--model_dir_path",
        type=str,
        required=True,
        help="The path to the target model directory.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the target dataset.",
    )
    parser.add_argument(
        "--github_username", type=str, required=True, help="Your GitHub username."
    )
    parser.add_argument(
        "--github_pat",
        type=str,
        required=True,
        help="Your GitHub personal access token (PAT).",
    )
    args = parser.parse_args()

    upload_ommx(
        model_dir_path=args.model_dir_path,
        dataset_name=args.dataset_name,
        github_username=args.github_username,
        github_pat=args.github_pat,
    )
