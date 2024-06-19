import os
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
# import librosa
import numpy as np
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from TTS.api import TTS


def get_asr_pipeline(
    model_name="openai/whisper-small"
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
    def get_instance(cls, model_name="openai/whisper-small"):
        """Create or return the already created ASR pipeline instance."""
        if not cls._instance:
            cls._instance = get_asr_pipeline(model_name)
        return cls._instance

def transcribe_audio(file_obj):
    # Get the ASR pipeline
    asr_pipeline = ASRPipelineSingleton.get_instance()
    
    # Transcribe the audio
    outputs = asr_pipeline(
        file_obj, # here
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )
    return outputs['text']

class TTSModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        """Create or return the already created TTS model instance"""
        if not cls._instance:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance = TTS(model_name).to(device)
        return cls._instance

def initialize_tts(model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
    """Initialize the TTS model"""
    return TTSModelSingleton.get_instance(model_name)

def synthesize_speech(text, output_path):
    """Generate speech from text and save to output_path"""
    tts = TTSModelSingleton.get_instance()
    tts.tts_to_file(text=text, speaker_wav="/app/dialog_graph/samples/sample.wav", language="en", file_path=output_path)

