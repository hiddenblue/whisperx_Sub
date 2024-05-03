import os
from http import HTTPStatus
import requests
import json

# local LLM server
# LLM abstract class
class LLM:
    def __init__(self, api_key, base_url, mode):

        if os.getenv("OPENAPI_API_KEY"):
            self.api_key = os.getenv("OPENAPI_API_KEY")
        else: self.api_key = api_key
        self.base_url = base_url
        self.history_limit = 10

        self.mode = "chat" if mode == "chat" else "completion"


    def completion_translate(self):
        pass

    def chat_translate(self,transcribe_result):
        pass

    def load(self):
        pass


class Ollama(LLM):
    """
    This a large language mode class for ollama local platform.
    The specific calling api reference to https://github.com/ollama/ollama/blob/main/docs/api.md

    """

    def __init__(self, model_name, api_key, base_url, mode, translate_prompt, system_prompt="" ):
        super().__init__(api_key, base_url)
        if self.mode == "chat":

            if not system_prompt:
                self.system_prompt = "you are a helpful subtitle translator"

            self.translate_prompt = translate_prompt
            self.data = {
                "model": f"{model_name}",
                "messages": [
                    {
                        "role": "system",
                        "content": f"{system_prompt}",
                    }
                ],
                "stream": False,
                "keepalive": "10m"
            }

    def chat_translate(self,transcribe_result):
        tranlation(transcribe_result, self.data, translate_prompt=self.translate_prompt)

    def completion_translate(self):
        pass


def tranlation(transcribe_result: list, data: dict, translate_prompt: str):
    count = 1

    for i in transcribe_result:
        if len(data["messages"]) > 30:
            data["messages"] = data["messages"][-5:]
        data["messages"].append({"role": "user", "content": translate_prompt + " " + i[2]})

        response = requests.post('http://localhost:11434/api/chat', json=data)

        print(count, end=" ")
        if response.status_code == HTTPStatus.OK:
            # print(json.loads(response.content)['message'], flush=True)
            print(json.loads(response.content)['message']["content"], flush=True)
            test = json.loads(response.content)['message']
            i[2] = json.loads(response.content)['message']["content"]
        else:
            print('Request count: %s, Status code: %s, error code: %s, error message: %s' % (
                count,  response.status_code, response.json(), response.content
            ))
            print(response.json(), flush=True)
        # data["messages"].append(test)
        count += 1
        # if count > limit:
        #     break









