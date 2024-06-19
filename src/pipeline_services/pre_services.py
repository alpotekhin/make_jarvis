"""
Pre Services
---
This module defines services that process user requests before script transition.
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"
from dff.script import Context

from qa.rag import retrieve
from qa.nlu import classify_message
from .messenger_interface import MessengerInterfaceSingleton
from .voice_processor import transcribe_audio


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def question_processor(ctx: Context):
    last_request = ctx.last_request
    messenger_interface = MessengerInterfaceSingleton.get_instance()
    
    if last_request is None:
        return
    
    if last_request.update.voice is not None:
        file_info = messenger_interface.messenger.get_file(last_request.update.voice.file_id)
        downloaded_file = messenger_interface.messenger.download_file(file_info.file_path)
        
        temp_file_path = os.path.join('/tmp', 'temp.wav')
        with open(temp_file_path, 'wb') as new_file:
            new_file.write(downloaded_file)
            
        last_request.text = transcribe_audio(temp_file_path)
        
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
