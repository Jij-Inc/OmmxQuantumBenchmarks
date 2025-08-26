from typing import Final

from ommx_quantum_benchmarks.uploader import Uploader as BaseUploader


class Uploader(BaseUploader):
    IMAGE_NAME: Final[str] = "qoblib"

    def push_ommx(
        self, image_tag: str, ommx_filepath: str, verification: bool = False
    ) -> None:
        """Push an OMMX archive file to GitHub Container Registry.

        Args:
            image_tag (str): an image tag for GitHub Container Registry.
            ommx_filepath (str): an OMMX archive file path.
            verification (bool, optional):
                if True, verify the experiment. Defaults to False.
                This functionality is primaliry for debugging purpose.
        """
        super().push_ommx(
            image_name=Uploader.IMAGE_NAME,
            image_tag=image_tag,
            ommx_filepath=ommx_filepath,
            verification=verification,
        )
