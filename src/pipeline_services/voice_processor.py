import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

def get_asr_pipeline(
    model_name="openai/whisper-tiny.en"
):
    """Set up the ASR pipeline with the specified model."""
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,  # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device="cuda:0",  # or mps for Mac devices
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )
    return pipe

class ASRPipelineSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, model_name="openai/whisper-tiny.en"):
        """Create or return the already created ASR pipeline instance."""
        if not cls._instance:
            cls._instance = get_asr_pipeline(model_name)
        return cls._instance

def transcribe_audio(file_obj):
    asr_pipeline = ASRPipelineSingleton.get_instance()
    outputs = asr_pipeline(
        file_obj,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )
    return outputs['text']

