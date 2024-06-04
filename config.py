
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
audio_file = "./Adding_Coredumps_to_Your_Debugging_Toolkit.aac"





# subtitle translation parameters
# if you want to  execute subtitle translation, you need to provide a LLM, local or remote

is_using_local_model = False

base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

api_key = ""

translation_model_name = "qwen-plus"

translation_prompt = ""

srt_file_name = "./output/cut/Adding_Coredumps_to_Your_Debugging_Toolkit.srt"



# rePunctuation parameters

# repunctuation_model = FullStopModel()

# split long function

WORDS_NUM_LIMITS = 12