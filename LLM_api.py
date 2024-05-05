import os
from http import HTTPStatus
import requests
import json
from openai import OpenAI


# local LLM server
# LLM abstract class
class LLM:
    def __init__(self, api_key, base_url, mode, model_name: str, translate_prompt):

        if os.getenv("OPENAPI_API_KEY"):
            self.api_key = os.getenv("OPENAPI_API_KEY")
        else:
            self.api_key = api_key
        self.base_url = base_url
        self.history_limit = 30

        self.mode = "chat" if mode == "chat" else "completion"
        self.model_name = model_name
        self.translate_prompt = translate_prompt

    def completion_translate(self):
        pass

    def chat_translate(self, transcribe_result):
        pass

    def load(self):
        pass

class Ollama(LLM):
    """
    This a large language mode class for ollama local platform.
    The specific calling api reference to https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    default_baseurl = "http://localhost:11434/api/chat"

    def __init__(self, model_name, api_key, translate_prompt, base_url=default_baseurl, mode="chat", system_prompt=""):
        super().__init__(api_key, base_url, mode, model_name, translate_prompt)
        if self.mode == "chat":

            if not system_prompt:
                self.system_prompt = "you are a helpful subtitle translator"

            self.data = {
                "model": f"{model_name}",
                "messages": [
                    {
                        "role": "system",
                        "content": f"{self.system_prompt}",
                    }
                ],
                "stream": False,
                "keepalive": "5m"
            }

    def chat_translate(self, transcribe_result):
        translation(transcribe_result,
                    data=self.data,
                    base_url=self.base_url,
                    translate_prompt=self.translate_prompt,
                    )

    def completion_translate(self):
        pass

    def close_model(self):
        """
        By sending a package with empty message or set the "keepalive" to "0".
        You can close the running model and retrieve the GPU memory
        :return:
        """
        self.data = {
            "model": f"{self.model_name}",
            "messages": [
                {
                    "role": "system",
                    "content": f"{self.system_prompt}",
                }
            ],
            "stream": False,
            "keepalive": "0"
        }


def translation(transcribe_result: list, data: dict, translate_prompt: str, base_url):
    count = 1

    for i in transcribe_result:
        if len(data["messages"]) > 30:
            data["messages"] = data["messages"][-5:]
        data["messages"].append({"role": "user", "content": translate_prompt + " " + i[2]})

        response = requests.post(f'{base_url}', json=data)

        print(count, end=" ")
        if response.status_code == HTTPStatus.OK:
            # print(json.loads(response.content)['message'], flush=True)
            content = json.loads(response.content)
            print(content['message']["content"], flush=True)
            current_answer = content['message']
            i[2] = content['message']["content"]
        else:
            print('Request count: %s, Status code: %s, error code: %s, error message: %s' % (
                count, response.status_code, response.json(), response.content
            ))
            print(response.json(), flush=True)
            current_answer = ""
        data["messages"].append(current_answer)
        count += 1
        # if count > limit:
        #     break
    # here we send a message that set keepalive to False


class OPENAI_General_Interface(LLM):
    """
    A lot of LLM platforms provide Openai-format compatible interface including ChatGLM Qwen and Ollama
    So you can call this interface to utilize a lot LLM
    help yourself.
    """

    def __init__(self, model_name: str, api_key, base_url, mode, translate_prompt, system_prompt=""):
        super().__init__(api_key, base_url, mode, model_name)

        # calling openai compatible interface
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.messages = {[
            {
                "role": "system",
                "content": f"{system_prompt}",
            }
        ]}

    def chat_translate(self, transcribe_result):
        count = 1

        for i in transcribe_result:
            if len(self.messages) > 10:
                self.messages = self.messages[-5:]
            self.messages.append(
                {
                    "role": "user",
                    "content": self.translate_prompt + " " + i[2]
                })
            message = ""

            try:
                completion = self.client.chat.completions.create(
                    messages=self.messages,
                    model=self.model_name,
                    max_tokens=1500,
                    stream=False
                )
                print(f"calling id: {completion.id}, model_name: {self.model_name}, token_usage: {completion.tokens}",
                      flush=True)
                print(f"input_tokens: {completion.tokens}, output_tokens: {completion.tokens}", flush=True)
                print(count, end=" ", flush=True)

                # the return message is different from the send messages
                message = completion.choices[0]["message"]
                print(message, flush=True)
                i[2] = message["content"]

            except Exception as e:
                print(f"Error: {e}")
                exit()

            finally:
                if not message:
                    self.messages.append({
                        "role": "assistant",
                        "content": ""
                    })
                else:
                    self.messages.append(message)

            count += 1
            # if count > limit:
            #     break
        # here we send a message that set keepalive to False

# def