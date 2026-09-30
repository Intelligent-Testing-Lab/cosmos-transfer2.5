import json
import logging
import os
import subprocess
from typing import Optional, Union


logger = logging.getLogger("cosmos")

DEBUG = True
IS_SILENT = False


class TransferInferenceObject:
    """
    Holds configuration for one or more Cosmos Transfer 2.5 invocations.

    Input paths may be supplied:
    - Directly through `input_paths`
    - Through INPUT_COSMOS_PATHS as a JSON list
    - Through the original INPUT_COSMOS_PATH for single-input compatibility

    Example INPUT_COSMOS_PATHS:
        [
          "/data/cosmos/job_001/generation_config.json",
          "/data/cosmos/job_002/generation_config.json"
        ]
    """

    def __init__(
        self,
        max_gpus: Optional[int] = None,
        master_port: Optional[int] = None,
        input_paths: Optional[list[str]] = None,
        output_path: Optional[str] = None,
    ) -> None:
        self.max_gpus = (
            max_gpus if max_gpus is not None else int(os.getenv("MAX_GPUS", "4"))
        )
        self.master_port = (
            master_port
            if master_port is not None
            else int(os.getenv("MASTER_PORT", "12341"))
        )

        self.input_paths = self._resolve_input_paths(input_paths)

        self.output_path = output_path or os.getenv(
            "OUTPUT_COSMOS_PATH",
            "/data/cosmos/cosmos-transfer2.5/outputs/"
            "138efbd6-9951-415e-aebe-2d9bc41524e3",
        )

        self._validate()

    @staticmethod
    def _parse_input_paths_env() -> Optional[list[str]]:
        """
        Parse INPUT_COSMOS_PATHS as JSON.

        Example:
        export INPUT_COSMOS_PATHS='[
          "/data/cosmos/input_001/generation_config.json",
          "/data/cosmos/input_002/generation_config.json"
        ]'
        """
        raw_input_paths = os.getenv("INPUT_COSMOS_PATHS")

        if not raw_input_paths:
            return None

        try:
            input_paths = json.loads(raw_input_paths)
        except json.JSONDecodeError as e:
            raise ValueError(
                "INPUT_COSMOS_PATHS must be a JSON list of input paths."
            ) from e

        if not isinstance(input_paths, list):
            raise ValueError(
                "INPUT_COSMOS_PATHS must decode to a JSON list, "
                'e.g. ["/path/a.json", "/path/b.json"].'
            )

        return input_paths

    def _resolve_input_paths(
        self,
        input_paths: Optional[list[str]],
    ) -> list[str]:
        """
        Resolution priority:
        1. input_paths passed to the constructor
        2. INPUT_COSMOS_PATHS JSON environment variable
        3. INPUT_COSMOS_PATH single-input environment variable
        """
        if input_paths is not None:
            return input_paths

        env_input_paths = self._parse_input_paths_env()
        if env_input_paths is not None:
            return env_input_paths

        return [
            os.getenv(
                "INPUT_COSMOS_PATH",
                "/data/cosmos/cosmos-transfer2.5/outputs/"
                "138efbd6-9951-415e-aebe-2d9bc41524e3/generation_config.json",
            )
        ]

    def _validate(self) -> None:
        if self.max_gpus < 1:
            raise ValueError(f"max_gpus must be >= 1, got {self.max_gpus}")

        if not self.input_paths:
            raise ValueError("At least one input path must be provided.")

        if not self.output_path:
            raise ValueError("An output path must be provided.")

        for input_path in self.input_paths:
            if not isinstance(input_path, str) or not input_path.strip():
                raise ValueError(f"Invalid input path: {input_path!r}")

    def _single_gpu_inference(self, input_path: list[str]) -> list[str]:
        prefix =  [ 
            "python",
            "examples/inference.py",
        ]
        suffix = [
             "-o",
            self.output_path,
            "--disable-guardrails"
        ]
        
        batch = []
        for path in input_path:
            batch.append("-i")
            batch.append(path)
        
        prefix.extend(batch)
        prefix.extend(suffix)
        
        return prefix

    def _multi_gpu_inference(self, input_path: list[str]) -> list[str]:
        prefix =  ["torchrun",
                    f"--nproc_per_node={self.max_gpus}",
                    f"--master_port={self.master_port}",
                    "examples/inference.py"
        ]
        suffix = [
             "-o",
            self.output_path,
            "--disable-guardrails"
        ]
        
        batch = ["-i"]
        for path in input_path:
            batch.append(path)
        prefix.extend(batch)
        prefix.extend(suffix)
                     
        return prefix

    def is_distributed(self) -> bool:
        return self.max_gpus > 1

    def get_cmd(self, input_path: list[str]) -> list[str]:
        if self.is_distributed():
            return self._multi_gpu_inference(input_path)

        return self._single_gpu_inference(input_path)


def _run_batch_input(
    inference_options: TransferInferenceObject
) -> None:
    """
    Run Cosmos Transfer for one generation_config.json input.
    """
    cmd = inference_options.get_cmd(inference_options.input_paths)

    logger.info("Inputs:")
    for batch in inference_options.input_paths:
        logger.info(" "*5 + batch)
    logger.info("Output directory: %s", inference_options.output_path)

    if DEBUG:
        logger.info("Executing Cosmos command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            env=os.environ.copy(),
            capture_output=IS_SILENT,
            text=True,
            check=True,
        )

        if IS_SILENT and result.stdout:
            logger.debug("Cosmos stdout:\n%s", result.stdout)

    except subprocess.CalledProcessError as e:
        logger.exception(
            "Cosmos Transfer failed for with return code=%d",
            e.returncode,
        )

        if IS_SILENT:
            if e.stdout:
                logger.error("Cosmos stdout:\n%s", e.stdout)
            if e.stderr:
                logger.error("Cosmos stderr:\n%s", e.stderr)


def _run(inference_options: TransferInferenceObject) -> None:
    """
    Run every supplied input as a batch with cosmos transfer.
    """
    total_inputs = len(inference_options.input_paths)

    logger.info(f"Starting batch job: {total_inputs} inputs")
    _run_batch_input(inference_options)
    logger.info(f"Completed job: {inference_options.input_paths}")

def main() -> None:
    logging.basicConfig(
        level=logging.INFO if DEBUG else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    inference_object = TransferInferenceObject()

    print(
        f"Using {inference_object.max_gpus} GPU(s) for "
        f"{len(inference_object.input_paths)} input(s)."
    )

    _run(inference_object)


if __name__ == "__main__":
    main()