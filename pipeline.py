from typing import Any, Union
from uuid import uuid4
import yaml
from pathlib import Path
import os
import subprocess
import random

class AbstractConfig():
    def set_key(self, key: str, item: Any) -> None:
        self.__setattr__(key, item)
        
    def get_dict(self) -> dict:

        data = dict()

        for key in iter(self.__dict__):
            
            if key not in self.__dict__:
                print(f"[ERROR] {key} not found, skipping")
                continue
            
            value = self.__dict__[key]
            if value is not None:
                if hasattr(value, 'get_dict'):
                    data[key] = value.get_dict()
                else:
                    data[key] = value
        return data
    
    def from_dict(self, data: dict):        
        for key in iter(data.keys()):
            
            if key not in self.__dict__:
                print(f"[ERROR] {key} not found, skipping")
                continue
            
            value = self.__dict__[key] 
            if value is not None:
                # check if its supposed to be an object
                if hasattr(value, 'get_dict'):
                    self.set_key(key, value.from_dict(data[key]))
                else:
                    self.set_key(key, data[key])
        return self
    
    def from_yaml(self, path: str="test_config.yml", mode: str="r") -> 'AbstractConfig':
        path_obj = Path(path)
        if not path_obj.exists():
            print("[WARNING] Input path does not exist. Returning default config")
            return self
        
        with open(path_obj, mode) as f:
            yaml_dict = yaml.safe_load(f)
        
        return self.from_dict(yaml_dict)
    
    def to_yaml(self, output: str="test_config.yml", mode: str="w") -> None:
        path_obj = Path(output)
        if path_obj.exists():
            print("[WARNING] Overrding existing config...")
            
        obj_dict = self.get_dict()
        
        with open(path_obj, mode) as f:
            yaml.safe_dump(obj_dict, f, default_flow_style=False, sort_keys=False)
        

class RecorderConfig(AbstractConfig):
    
    def __init__(self) -> None:        
        self.scenario_length=20
        self.fixed_delta=0.05
        self.traffic=True    
        self.vehicles=50
        self.traffic_speed_differential=10
        self.town="Town10HD"
        self.address="localhost"
        self.port=2000

class ControlConfig(AbstractConfig):
    
    def __init__(self, name: str, enabled: bool=False) -> None:
        self.name = name
        
        self.enabled=enabled
        self.control_path=""
        self.mask_path=""
        self.weight=0.0
    

class TransferConfig(AbstractConfig):
    def __init__(self) -> None:
        self.input_rgb=""
        self.prompt_path=""
        self.guidance=3
        self.seg=ControlConfig(name="seg")
        self.edge=ControlConfig(name="edge")
        self.depth=ControlConfig(name="depth")
        self.vis=ControlConfig(name="vis")

# this entire config gets passed to each object thats called - each object represents a stage of the pipeline
class PipeplineConfig(AbstractConfig):
    def __init__(self) -> None:
        self.runs=5
        self.run_index = 0
        self.generator_seed=10
        self.output_dir = "./outputs"
        
        # scenario recorder (replace with leaderboard at one point)
        self.scenario_recorder_config = RecorderConfig()
        self.transfer_config = TransferConfig()
       
    def _new_run_uuid(self) -> None:
        path_obj = Path(self.output_dir) / str(uuid4())
        self.run_output_dir = str(path_obj.resolve())
        
    def get_run_index(self) -> int:
        return self.run_index
    
    def next_run(self) -> None:
        if self.run_index < self.runs:
            self.run_index += 1
            self._new_run_uuid()       
            
    def build_path(self, tree: list[str]) -> str:
        """
        Returns a complete path from the self.run_output_dir
        """
        path_obj = Path(self.run_output_dir)
        for paths in tree:
            path_obj = path_obj / paths
        return str(path_obj.resolve())

class PipelineStage(object):
    """
    Abstract class - each stage will inherit these methods and they must be populated
    """
    IS_SILENT=False
    DEBUG=True
    
    def __init__(self, pipeline_stage: str, config: PipeplineConfig) -> None:
        self.config = config
        self.pipeline_stage = pipeline_stage
    
    def _run(self, cmd: list[str], env: Union[dict, os._Environ[str]], as_shell: bool=False) -> None:
        for index, item in enumerate(cmd):
            if not isinstance(item, str) or not isinstance(item, bytes):
                cmd[index] = str(item)      
        try:
            if self.DEBUG:
                print(f"[INFO] Executing stage: {self.pipeline_stage}")
                
            if as_shell:
                cmd = " ".join(cmd) # type: ignore
            
            proc = subprocess.run(
                cmd,
                env=env,  
                shell=as_shell,
                executable="/bin/bash",
                capture_output=self.IS_SILENT,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print("[ERROR] Failed executing pipeline stage")#
    
    def _random(self) -> int:
        return random.randint(0, 100000)
    
    def run_stage(self) -> None:
        raise NotImplementedError
        
    def build_env(self ) -> Union[dict, os._Environ[str]]:
        """
        Map the config elements to env variables used by the scripts. By default returns os.environ
        """
        return os.environ
    
class RecorderStage(PipelineStage):
    ENV_PATH="/data/cosmos/cosmos-transfer2.5/prompt_processor/.venv/bin/python3"
    SCRIPT_PATH="/data/cosmos/cosmos-transfer2.5/basic_recorder.py"
    
    def __init__(self, config: PipeplineConfig) -> None:
        super().__init__("recorder", config)
    
    def run_stage(self) -> None:
        env = self.build_env()
        cmd = [
            self.ENV_PATH, self.SCRIPT_PATH,
            "--town", self.config.scenario_recorder_config.town,
            "--seconds", self.config.scenario_recorder_config.scenario_length,
            "--delta", self.config.scenario_recorder_config.fixed_delta,
            "--seed", self._random(),
            "--output-dir", self.config.build_path(
                ["sensor_recordings"]
            )
        ]

        if self.config.scenario_recorder_config.traffic:
            cmd.extend([
                "--traffic-vehicles", self.config.scenario_recorder_config.vehicles,
                "--traffic-speed-difference", self.config.scenario_recorder_config.traffic_speed_differential,
            ])
        
        print(cmd)
        self._run(cmd, env, as_shell=True)
    
    def build_env(self) -> Union[dict, os._Environ[str]]:
        return super().build_env()
    
    
class PreProcessorStage(PipelineStage):
    ENV_PATH="/data/cosmos/cosmos-transfer2.5/.venv/bin/python3"
        
    def __init__(self, config: PipeplineConfig) -> None:
        super().__init__("preprocessor", config)
        
    def build_env(self) -> Union[dict, os._Environ[str]]:
        env_dict = os.environ.copy()
        
        # inputs
        env_dict["RGB_INPUT"]=self.config.build_path(["sensor_recordings","front_rgb","front_rgb.mp4"])
        env_dict["SEMANTIC_INPUT"] = self.config.build_path(["sensor_recordings", "front_semantic_cityscapes", "front_semantic_cityscapes.mp4"])
        env_dict["DEPTH_INPUT"] = self.config.build_path(["sensor_recordings", "front_depth_log", "front_depth_log.mp4"])                                              
        env_dict["RAW_SEMANTIC_INPUT"] = self.config.build_path(["sensor_recordings", "front_semantic_id", "raw_frames.mp4"])

        # outputs
    

        return env_dict
    
if __name__ == "__main__":
    # load config
    config = PipeplineConfig()
    config.from_yaml()
    
    # initialise values
    random.seed(int(config.generator_seed))
    config.next_run() # initialise next run
    
    RecorderStage(config).run_stage()
    
        
    
    
    

    
    