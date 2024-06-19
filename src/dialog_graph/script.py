"""
Script
--------
This module defines a script that the bot follows during conversation.
"""

import os
from dff.script import RESPONSE, TRANSITIONS, LOCAL
import dff.script.conditions as cnd
from dff.messengers.telegram import TelegramMessage
from dff.script.core.message import Attachments, Audio

from .responses import answer_question
from .conditions import received_statement, received_question

script = {
    "service_flow": {
        "start_node": {
            TRANSITIONS: {
                ("qa_flow", "welcome_node"): cnd.exact_match(
                    TelegramMessage(text="/start")
                )
            },
        },
        "fallback_node": {
            RESPONSE: TelegramMessage(
                text="Something went wrong. Use `/restart` to start over."
            ),
            TRANSITIONS: {
                ("qa_flow", "welcome_node"): cnd.exact_match(
                    TelegramMessage(text="/restart")
                )
            },
        },
    },
    "qa_flow": {
        LOCAL: {
            TRANSITIONS: {
                ("qa_flow", "answer_question"): received_question,
                ("qa_flow", "welcome_node"): cnd.exact_match(
                    TelegramMessage(text="/restart")
                ),
                ("qa_flow", "chitchat_node"): received_statement,
            },
        },
        "chitchat_node": {
            RESPONSE: TelegramMessage(
                attachments=Attachments(files=[Audio(source="/app/dialog_graph/samples/idk.ogg")])
            ),
        },
        "welcome_node": {
            RESPONSE: TelegramMessage(
                attachments=Attachments(files=[Audio(source="/app/dialog_graph/samples/welcome.ogg")])
            ),
        },
        "answer_question": {
            RESPONSE: answer_question,
        },
    },
}
