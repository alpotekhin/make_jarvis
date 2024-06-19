"""
Pre Services
---
This module defines services that process user requests before script transition.
"""

from dff.script import Context

from qa.rag import retrieve
from qa.nlu import classify_message
from .messenger_interface import MessengerInterfaceSingleton
from .voice_processor import transcribe_audio


import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def question_processor(ctx: Context):
    last_request = ctx.last_request
    logger.info(f"GET THE QUERY: {last_request}")
    
    messenger_interface = MessengerInterfaceSingleton.get_instance()
    
    if last_request is None:
        return
    
    logger.info(f"CHECK VOICE: {last_request.update.voice}")
    
    if last_request.update.voice is not None:
        file_info = messenger_interface.messenger.get_file(last_request.update.voice.file_id)
        downloaded_file = messenger_interface.messenger.download_file(file_info.file_path)
        
        # Use BytesIO to handle file operations in memory
        with io.BytesIO(downloaded_file) as audio_file:
            # You might need to adjust this if your ASR pipeline expects a file path.
            last_request.text = transcribe_audio(audio_file)
        
    logger.info(f"last_request.text: {last_request.text}")
    
    if last_request.annotations is None:
        last_request.annotations = {}
    else:
        if last_request.annotations.get(
            "retrieved_docs"
        ) and last_request.annotations.get("intent"):
            return
        
    if last_request.text is None:
        last_request.annotations["retrieved_docs"] = None
        last_request.annotations["intent"] = None
    else:
        last_request.annotations["retrieved_docs"] = retrieve(last_request.text)
        last_request.annotations["intent"] = classify_message(last_request.text)

    ctx.add_request(last_request)



services = [question_processor]  # pre-services run before bot sends a response
