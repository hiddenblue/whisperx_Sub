
# task type
# task could be "transcribe" or "all"

task = "transcribe"


# output and temp directory path

output_dir = "./output"

temp_dir = "./temp"

output_format = "all"


# transcribe parameters


# 需要你配置的一些变量
# 音频或者视频文件路径
audio_file = "./openai_sample.mp3"





# subtitle translation parameters
# if you wanna execute subtitle translation, you need to provide a LLM, local or remote

is_using_local_model = False

base_url = "http://localhost:11434/api/chat"

translation_model_name = "qwen:32b-chat-v1.5-q5_K_M"

translation_prompt = ""

srt_file_name = ""



# rePunctuation parameters

# repunctuation_model = FullStopModel()

# split long function

WORDS_NUM_LIMITS = 12