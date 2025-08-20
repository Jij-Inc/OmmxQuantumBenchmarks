from pathlib import Path


def load_topology_instance(instance_path: str | Path) -> dict[str, object]:
    """Load a Topology instance from the given directory.

    Args:
        instance_path (str | Path): Path to the instance directory or .dat file

    Raises:
        FileNotFoundError: If the path does not exist or is not a valid .dat file
        ValueError: If the .dat file format is invalid

    Returns:
        dict[str, object]: Dictionary containing all the data needed for JijModeling
    """
    path = Path(instance_path)

    # If path is a .dat file, use it directly
    if path.is_file() and path.suffix == ".dat":
        dat_file = path
    else:
        raise FileNotFoundError(
            f"Path does not exist or is not a valid .dat file: {path}"
        )

    # Read the .dat file
    with open(dat_file, "r") as f:
        content = f.read().strip()

    # Parse the content: format is "nodes degree"
    parts = content.split()
    if len(parts) != 2:
        raise ValueError(
            f"Invalid .dat file format. Expected 'nodes degree', got: {content}"
        )

    # Extract nodes and degree
    nodes = int(parts[0])
    degree = int(parts[1])

    return {"nodes": nodes, "degree": degree}
