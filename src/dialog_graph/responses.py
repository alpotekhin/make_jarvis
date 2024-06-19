"""
Responses
---------
This module defines different responses the bot gives.
"""

from typing import cast

from dff.script import Context
from dff.pipeline import Pipeline
from dff.messengers.telegram import TelegramMessage
from dff.script.core.message import Attachments, Audio
from qa.rag import format_document, generate
from pipeline_services.voice_processor import synthesize_speech


def answer_question(ctx: Context, _: Pipeline):
    """Answer a question asked by a user by pressing a button."""
    if (
        ctx.validation
    ):  # this function requires non-empty fields and cannot be used during script validation
        return TelegramMessage()
    last_request = ctx.last_request
    if last_request is None:
        raise RuntimeError("No last requests.")
    last_request = cast(TelegramMessage, ctx.last_request)
    # if last_request.callback_query is None:
    #     raise RuntimeError("No callback query")
    retrieved_docs = ctx.last_request.annotations.get("retrieved_docs")
    context = "\n".join(
        [format_document(doc, i + 1) for i, doc in enumerate(retrieved_docs)]
    )
    gen_answer = generate(question=last_request.text, context=context)
    synthesize_speech(gen_answer, "/tmp/temp_ans.ogg")

    return TelegramMessage(attachments=Attachments(files=[Audio(source="/tmp/temp_ans.ogg")]))

