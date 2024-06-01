import os
from http import HTTPStatus
import requests
import json
from openai import OpenAI
import time
from abc import ABC, abstractmethod
import copy
from typing import List, Dict, Tuple


# local LLM server
# LLM abstract class
class LLM(ABC):
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

    def chat_translate(self, transcribe_result, translate_prompt):
        pass

    def load(self):
        pass

    def batch_translate(self, transcribe_result):
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
                self.system_prompt = "you are a professional subtitle translator"
                # self.translate_prompt=""

            self.data = {
                "model": f"{model_name}",
                "messages": [
                    {
                        "role": "system",
                        "content": f"{self.system_prompt}",
                    }
                ],
                "stream": False,
                "keepalive": "5m",
                # "options": {
                #     # "seed": 1000
                #     "temperature": 0.4
                # }
            }

    def chat_translate(self, transcribe_result, translate_prompt):
        translation(transcribe_result,
                    data=self.data,
                    base_url=self.base_url,
                    translate_prompt=translate_prompt,
                    )

    def completion_translate(self):
        pass

    @staticmethod
    def close_model(model_name, baseurl):
        """
        By sending a package with empty message or set the "keepalive" to "0".
        You can close the running model and retrieve the GPU memory
        :return:
        """
        data = {"model": model_name, "keep_alive": 0}
        response = requests.post(baseurl, json=data)
        print(response.content)
        time.sleep(1)

    # 32b model is unable to perform batch translation for unstable output format
    def batch_translate(self, transcribe_result: list):
        data = self.data
        translate_prompt = self.translate_prompt
        base_url = self.base_url

        translate_prompt= "Please translate the following continuous 10 lines of English subtitles into Chinese, without any explanations. One sentence at each line"
        # each we translate 10 line at once
        for i in range(0, len(transcribe_result), 10):
            tmp = ""
            if len(data["messages"]) > 5:
                data["messages"] = data["messages"][-2:]

            sentences_list = [transcribe_result[j][2] for j in range(i, i+10)]
            sentences_list = MessageFormatter.format_message_list(sentences_list)

            tmp = "\n".join(sentences_list)
            data["messages"].append({"role": "user", "content": translate_prompt + "\n" + tmp})

            response = requests.post(f'{base_url}', json=data)
            print("count: ", i)

            content = json.loads(response.content)
            if response.status_code == 200:
                print(response.status_code)
                message = content["message"]
                answer = message["content"]
                print(answer)
                try:
                    ret_list = content["message"]["content"].split("\n\n")
                    if len(ret_list) == 10:
                        for j in range(len(ret_list)):
                            transcribe_result[i + j][2] = ret_list[j]
                except KeyError as e:
                    print(e)
                finally:
                    data["messages"].append(message)
            else:
                print(response.status_code)
                message = content["message"]
                data["messages"].append(message)
                continue


class   MessageFormatter:

    def format_message_list(message_list: List)-> List:
        """

        format the message to facilitate LLM translation, incluing captinlize the first alphabet.
        add . or  ? to the end of each sentence.
        :param message_list:
        :return:
        """

        for i in range(len(message_list)):
            message_list[i] = message_list[i].strip()
            message_list[i] = message_list[i][0].capitalize() + message_list[i][1:]
            # if (message_list[i][-1] == ","
            #         not in [".", "?", "!"]):
            #     message_list[i] += "."
        return message_list


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
        super().__init__(api_key, base_url, mode="chat", model_name=model_name, translate_prompt=translate_prompt)
        print("Your model_name is ", model_name)
        if not system_prompt:
            self.system_prompt = "you are a professional subtitle translator"
            # self.translate_prompt=""

        self.messages = [
                {
                    "role": "system",
                    "content": f"{self.system_prompt}",
                }
            ]
        # calling openai compatible interface
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat_translate(self, transcribe_result, translate_prompt):
        messages = copy.deepcopy(self.messages)

        if not translate_prompt:
            translate_prompt = "Translate this English sentence into Chinese."

            for i in range(len(transcribe_result)):
                if len(messages) > 10:
                    messages = messages[-4:]

                tmp = MessageFormatter.format_message_list([transcribe_result[i][2]])[0]
                messages.append(
                    {
                        "role": "user",
                        "content": translate_prompt + " " + tmp
                    })

                try:
                    resp = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        stream=False
                    )
                    print("count: ", i)
                    print(f"input_tokens: {resp.usage.prompt_tokens}, output_tokens: {resp.usage.completion_tokens}, total_tokens: {resp.usage.total_tokens}", flush=True)
                    message = resp.choices[0].message.content
                    print(message, flush=True)
                    transcribe_result[i][2] = message

                except Exception as e:
                    print(f"Error: {e}")
                    message = " "
                finally:
                    messages.append({
                        "role": "assistant",
                        "content": message
                    })

        # here we send a message that set keepalive to False
# def messge_constructor(batch_size:int, preserve_length:int, ):
    def batch_translate(self, transcribe_result: list):
        messages = copy.deepcopy(self.messages)
        translate_prompt = self.translate_prompt
        base_url = self.base_url
        client = self.client

        fail_list = []

        translate_prompt= "Please translate the following continuous 10 lines of English subtitles into Chinese, without any explanations. One sentence at each line."
        # each we translate 10 line at once
        for i in range(0, len(transcribe_result), 10):
            tmp = ""
            if len(messages) > 4:
                messages = messages[-2:]
            if (i+10) > len(transcribe_result):
                sentences_list = [transcribe_result[j][2] for j in range(i, len(transcribe_result))]
            else:
                sentences_list = [transcribe_result[j][2] for j in range(i, i+10)]

            sentences_list = MessageFormatter.format_message_list(sentences_list)

            tmp = "\n".join(sentences_list)
            messages.append({"role": "user", "content": translate_prompt + "\n" + tmp})

            try:
                resp = client.chat.completions.create(
                    messages=messages,
                    model= self.model_name,
                    stream=False)

                # the return resp is a namedtuple, like this
                """
                ChatCompletion(id='chatcmpl-93d159a2a94a9e90a54822ec01d9f5d0', choices=[Choice(finish_reason='stop', index=0, logprobs=None, 
                message=ChatCompletionMessage(content='我正在导入OpenAI库。', role='assistant', function_call=None, tool_calls=None))],
                created=1716647537, model='qwen1.5-110b-chat', object='chat.completion', system_fingerprint=None,
                usage=CompletionUsage(completion_tokens=149, prompt_tokens=241, total_tokens=390))
                """

                print("count: ", i)
                print(f"input_tokens: {resp.usage.prompt_tokens}, output_tokens: {resp.usage.completion_tokens}, total_tokens: {resp.usage.total_tokens}", flush=True)
                message = resp.choices[0].message.content
                print(message, "\n")
                ret_list = (message.split("\n"))
                print("length of ret_list", len(ret_list))
                if len(ret_list) == len(sentences_list):
                    for j in range(len(ret_list)):
                        transcribe_result[i + j][2] = ret_list[j]
                else:
                    fail_list.append(i)
            except Exception as e:
                # print the exception with red color
                print(e)
                message = " "
                fail_list.append(i)
            finally:
                messages.append({"role": "assistant", "content": message})

        if fail_list:
            remain_transcribe_result = []

            for i in fail_list:
                if i < len(transcribe_result)-10:
                    remain_transcribe_result += transcribe_result[i:i+10]
                if fail_list[-1]+10 > len(transcribe_result):
                    remain_transcribe_result.append(transcribe_result[fail_list[-1]:])

            self.chat_translate(remain_transcribe_result, translate_prompt="")

            for index, value in enumerate(remain_transcribe_result):
                transcribe_result[int(value[0])] = value

        return transcribe_result

if __name__ == '__main__':
    
    from srt_util import srt_reader
    from srt_util import srt_writer
    from config import translation_model_name, base_url, srt_file_name, is_using_local_model


    if not translation_model_name:
        print("Please set the translation_model_name in config.py")
        exit(0)

    if not base_url:
        print("Please set the base_url in config.py")
        exit(0)

    if not srt_file_name:
        print("Please set the srt_file_name in config.py")
        exit(0)

    print(f"translation_model_name: {translation_model_name}, base_url: {base_url}, srt_file_name: {srt_file_name}")
    print("Start translating...")

    srt_content = srt_reader(srt_file_name)

    if is_using_local_model:

        # initial your LLM
        ollama = Ollama(model_name=translation_model_name,
                        api_key="",
                        base_url=base_url,
                        mode="chat",
                        translate_prompt="Translate this English sentence into Chinese. Keep the puncutaion if possible.")

        ollama.batch_translate(srt_content)

    else:
        # base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        API_KEY = ""

        openai_model = OPENAI_General_Interface(model_name=translation_model_name,
                                                api_key=API_KEY,
                                                base_url=base_url,
                                                mode="chat",
                                                translate_prompt="",
                                                system_prompt=""
                                                )

        openai_model.batch_translate(srt_content)
        # openai_model.chat_translate(srt_content, translate_prompt="")

    print([i[2] for i in srt_content])
    output_name = srt_file_name.split(".")[0] + "-zh-CN"+ ".srt"
    srt_writer(srt_content, output_name)

