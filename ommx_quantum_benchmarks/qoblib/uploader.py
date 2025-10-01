from ommx_quantum_benchmarks.uploader import Uploader as BaseUploader
from ommx_quantum_benchmarks.qoblib.definitions import IMAGE_NAME, get_instance_tag


class Uploader(BaseUploader):

    def push_ommx(
        self,
        dataset_name: str,
        model_name: str,
        instance_name: str,
        ommx_filepath: str,
        verification: bool = False,
    ) -> None:
        """Push an OMMX archive file to GitHub Container Registry.

        Args:
            dataset_name (str): a dataset name.
            model_name (str): a model name.
            instance_name (str): an instance name.
            ommx_filepath (str): an OMMX archive file path.
            verification (bool, optional):
                if True, verify the experiment. Defaults to False.
                This functionality is primarily for debugging purpose.
        """
        instance_tag = get_instance_tag(dataset_name, model_name, instance_name)
        super().push_ommx(
            image_name=IMAGE_NAME,
            image_tag=instance_tag,
            ommx_filepath=ommx_filepath,
            verification=verification,
        )
