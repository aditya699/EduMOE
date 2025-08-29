import importlib
from typing import Optional, Literal
from pydantic import BaseModel

# --- Safe optional imports ---
if importlib.util.find_spec("torch"):
    import torch
else:
    torch = None

if importlib.util.find_spec("datasets"):
    import datasets
else:
    datasets = None

if importlib.util.find_spec("transformers"):
    import transformers
else:
    transformers = None


class LibraryVersions(BaseModel):
    torch: str = " Not Installed"
    datasets: str = " Not Installed"
    transformers: str = " Not Installed"


class GPUInfo(BaseModel):
    device: Literal["GPU", "CPU", "No Device Detected"]
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    memory_gb: Optional[float] = None
    message: Optional[str] = None


class EnvironmentCheck:
    """Environment checker with Pydantic models."""

    def __init__(self) -> None:
        self.torch_version = torch.__version__ if torch else None
        self.datasets_version = datasets.__version__ if datasets else None
        self.transformers_version = transformers.__version__ if transformers else None

    def get_versions(self) -> LibraryVersions:
        """Return installed library versions as a Pydantic model."""
        return LibraryVersions(
            torch=self.torch_version or " Not Installed",
            datasets=self.datasets_version or " Not Installed",
            transformers=self.transformers_version or " Not Installed",
        )

    def check_gpu(self) -> GPUInfo:
        """Return GPU details as a Pydantic model."""
        if not torch:
            return GPUInfo(device="No Device Detected")

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GPUInfo(
                device="GPU",
                cuda_version=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(0),
                memory_gb=round(props.total_memory / (1024**3), 2),
            )
        return GPUInfo(device="CPU", message="No GPU detected")


# --- Usage for manual run ---
if __name__ == "__main__":
    env = EnvironmentCheck()
    print("Library Versions:", env.get_versions().model_dump())
    print("Device Info:", env.check_gpu().model_dump())
