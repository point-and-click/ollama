from os import path
from time import sleep

import ollama
from ollama import RequestError, ResponseError

import ai
from play.rules import RuleType
from settings.settings import Settings
from utils.logging import log

settings = Settings(path.join(path.split(path.relpath(__file__))[0], 'settings.yaml'))


class Chat:
    client = ollama.Client(host=settings.get("host"), timeout=settings.get("timeout"))

    @staticmethod
    def send(prompt, session, character=None):
        messages = []
        if character:
            messages.append(
                {"role": ai.Role.SYSTEM.value, "content": character.personality.get('description')}
            )
            if character.task:
                messages.append(
                    {"role": ai.Role.USER.value,
                     "content": f'{character.task.description} {character.serialize_rules(RuleType.PERMANENT)}'}
                )
        if session.history.summary:
            messages.append(session.history.summary.serialize())
        if session.history.moments:
            messages.extend(
                [entry.serialize(character.name) for entry in session.history.get()]
            )
        messages.append(
            {
                "role": ai.Role.USER.value,
                "content": f'{character.serialize_rules(RuleType.TEMPORARY)} {prompt}'
            }
        )

        model = character.personality.get("model", settings.get("chat.model"))

        try:
            completion = Chat.client.chat(
                model=model,
                messages=messages,
            )
        except (RequestError, ResponseError) as error:
            log.error(error)
            return

        try:
            return completion.get("message", {}).get("content", "")
        except IndexError as error:
            log.warning(error)
            return
