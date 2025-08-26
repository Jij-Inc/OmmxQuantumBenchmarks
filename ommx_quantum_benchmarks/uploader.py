import os
from pathlib import Path
from typing import Final

import minto
import ommx.v1
from ommx.artifact import Artifact


class Uploader:
    ORG: Final[str] = "Jij-Inc"
    REPO: Final[str] = "OMMX-OBLIB"

    def __init__(self) -> None:
        os.environ["OMMX_BASIC_AUTH_DOMAIN"] = "ghcr.io"

    def push_ommx(
        self,
        image_name: str,
        image_tag: str,
        ommx_filepath: str,
        verification: bool = False,
    ) -> None:
        """Push an OMMX archive file to GitHub Container Registry.

        Args:
            image_name (str): an image name for GitHub Container Registry.
            image_tag (str): an image tag for GitHub Container Registry.
            ommx_filepath (str): an OMMX archive file path.
            verification (bool, optional):
                if True, verify the experiment. Defaults to False.
                This functionality is primaliry for debugging purpose.

        Raises:
            ValueError: if the loaded experiment is invalid
        """
        # Load an instance data from an OMMX archive file.
        artifact = Artifact.load_archive(ommx_filepath)
        # Create an experiment and log the instance and solution data.
        experiment = minto.Experiment(name=image_name, auto_saving=False)
        key_name = Path(ommx_filepath).stem
        if artifact.instance is not None:
            experiment.log_instance(instance_name=key_name, instance=artifact.instance)
        else:
            raise ValueError(f"Instance is None for file: {ommx_filepath}")
        if artifact.solution is not None:
            experiment.log_solution(solution_name=key_name, solution=artifact.solution)
        else:
            print(f"Warning: Solution is None for file: {ommx_filepath}")

        # Verify the experiment.
        if verification:
            Uploader.veryfy_experiment(experiment, ommx_filepath)

        artifact = experiment.push_github(
            org=self.ORG, repo=self.REPO, name=image_name, tag=image_tag
        )

    @staticmethod
    def veryfy_experiment(experiment: minto.Experiment, ommx_filepath: str) -> None:
        """Verify the experiment by reloading the instance and solution data.

        Args:
            experiment (minto.Experiment): an experiment to be verified.
            ommx_filepath (str): an OMMX archive file path.

        Raises:
            ValueError: if the loaded instance or solution is invalid
        """
        # For now, expeiment must have only one instance.
        # Raise ValueError if the number of instances is not one.
        instances = experiment.get_current_datastore().instances
        if len(instances) != 1:
            raise ValueError(
                f"Number of instances in the given experiment is not one: {len(instances)}."
            )
        # Get the only instance in the experiment.
        name, instance = list(instances.items())[0]
        # Load an instance data from an OMMX archive file.
        loaded_artifact = Artifact.load_archive(ommx_filepath)
        loaded_instance = loaded_artifact.instance
        # Verify if those instances are same.
        if not Uploader.are_same_instances(instance, loaded_instance):
            raise ValueError(
                "The instance in the experiment is different from the loaded one."
            )

        # For now, expeiment must have at most one solution.
        solutions = experiment.get_current_datastore().solutions
        if len(solutions) > 1:
            raise ValueError(
                f"Number of solutions in the given experiment is more than one: {len(solutions)}."
            )
        # Get the only solution in the experiment if it exists.
        solution = solutions[name] if len(solutions) == 1 else None
        # Load a solution data from an OMMX archive file.
        loaded_solution = loaded_artifact.solution
        # Veryfy if those solutions are same.
        if solution is None and loaded_solution is None:
            pass
        elif solution is None and loaded_solution is not None:
            raise ValueError(
                "Solution in the experiment is None, but loaded solution is not None."
            )
        elif solution is not None and loaded_solution is None:
            raise ValueError(
                "Solution in the experiment is not None, but loaded solution is None."
            )
        else:  # solution and loaded_solution are not None.
            if not Uploader.are_same_solutions(solution, loaded_solution):
                raise ValueError(
                    "The solution in the experiment is different from the loaded one."
                )

    @staticmethod
    def are_same_instances(
        instance1: ommx.v1.Instance, instance2: ommx.v1.Instance
    ) -> bool:
        """Check if two ommx instances are the same.

        Args:
            instance1 (ommx.v1.Instance): the first ommx instance.
            instance2 (ommx.v1.Instance): the second ommx instance.

        Returns:
            bool: True if the instances are the same (decision variables, constraints, and objective match), False otherwise.
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

    @staticmethod
    def are_same_solutions(
        solution1: ommx.v1.Solution, solution2: ommx.v1.Solution
    ) -> bool:
        """Check if two OMMX solutions are equivalent.

        Args:
            solution1 (ommx.v1.Solution): The first solution to compare.
            solution2 (ommx.v1.Solution): The second solution to compare.

        Returns:
            bool: True if the solutions are the same (objective value, decision variables, and feasibility match), False otherwise.
        """
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
