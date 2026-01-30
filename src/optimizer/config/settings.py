from pydantic_settings import BaseSettings

class PipelineConfig(BaseSettings):
    batch_size: int = 50
    verifier_timeout_seconds: int = 300
    mcts_c_constant: float = 1.0
    llm_model_name: str = "claude-opus-4-5-20251101"
    cuda_home: str = "/usr/local/cuda-12.1"
    retry_limit: int = 3
    ancestor_code_depth: int = 3  # How many parent kernels to include in LLM prompt
    mcts_iterations: int = 5  # Number of optimization iterations per operator
    
    class Config:
        env_prefix = "OPTIMIZER_"

settings = PipelineConfig()
