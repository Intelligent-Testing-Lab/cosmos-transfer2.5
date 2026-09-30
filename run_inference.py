import subprocess
import os
import sys
import logging
import uuid
from typing import Union

logger = logging.getLogger("cosmos")
DEBUG = True
IS_SILENT = False

class TransferInferenceObject(object):
    """
    Global object to hold inference details of a single invocation of the CosmosTranfer model
    Accepts constructor overrides or setting via environmental variables
    """
    def __init__(
        self,
        max_gpus: Union[int, None] = None,
        master_port: Union[int, None] = None,
        input_path: Union[str, None] = None,
        output_path: Union[str, None] = None
    ) -> None:
        self.max_gpus = max_gpus if max_gpus is not None else int(os.getenv("MAX_GPUS", 4))
        self.master_port = master_port if master_port is not None else int(os.getenv("MASTER_PORT", 12341))
        self.input_path = input_path if input_path is not None else os.getenv("INPUT_COSMOS_PATH", "")
        self.output_path = output_path if output_path is not None else os.getenv("OUTPUT_COSMOS_PATH", "")

    def _single_gpu_inference(self) -> list[str]:
        return [
            "python",
            "examples/inference.py",
            "-i",
            self.input_path,
            "-o",
            self.output_path,
            "--disable-guardrails"
        ]

    def _multi_gpu_inference(self) -> list[str]:
        return [   
            "torchrun",
            f"--nproc_per_node={self.max_gpus}",
            f"--master_port={self.master_port}",
            "examples/inference.py",
            "-i",
            self.input_path,
            "-o",
            self.output_path,
            "--disable-guardrails"
        ]
        
    def is_distributed(self) -> bool:
        return self.max_gpus > 1
    
    def get_cmd(self) -> list[str]:
        return self._multi_gpu_inference() if self.is_distributed() else self._single_gpu_inference()

def _run(inference_options: TransferInferenceObject):
    """
    Runs CosmosTransfer 2.5 on a selected input
    """    
    cmd = inference_options.get_cmd()
    
    try:
        if DEBUG:
            logger.info(f"Executing cosmos command: {cmd}")
        proc = subprocess.run(
            cmd,
            env=os.environ,  # ensure that we inherit current env; this also happens by default
            capture_output=IS_SILENT,
            check=True,  # check that when cosmos transfer suceeds the return code is correct
        )
    except subprocess.CalledProcessError as e:
        logger.error("Failed executing cosmos transfer")


def main():
    inference_object = TransferInferenceObject()
    print(f"Using {inference_object.max_gpus} gpus")
    _run(inference_object)
    

if __name__ == "__main__":
    main()
