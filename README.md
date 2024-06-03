# whisperx_Sub

![GitHub stars](https://img.shields.io/github/stars/hiddenblue/whisperx_Sub?style=social)
![GitHub forks](https://img.shields.io/github/forks/hiddenblue/whisperx_Sub?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/hiddenblue/whisperx_Sub?style=social)
![GitHub repo size](https://img.shields.io/github/repo-size/hiddenblue/whisperx_Sub?)
![GitHub last commit](https://img.shields.io/github/last-commit/hiddenblue/whisperx_Sub?color=red)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhiddenblue%2Fwhisperx_Sub&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

Whisperx_sub 是一个基于Whisperx的视频听写和翻译的字幕生成工具

## 特性：

- 凭借[Faster whisper](https://github.com/guillaumekln/faster-whisper)语音模型,能够最多60x于原版whisper的速度生成英文字幕, 30分钟视频只需要1-3分钟就能生成完整的英文字幕

- [Whisperx](https://github.com/m-bain/whisperX) 作者利用wav2vec模型分别解决了whisper的幻听和单词停写不够准确的问题，改进后可以实现单词在逐秒尺度上的准确

- 总结英文断句规律和nltk等自然语言工具，实现了对较长英文句子的自动断句，可以在不影响翻译的情况下，达到对70%的英文长句的准确断句，大幅减少后期打轴的工作量

- 使用常见的[Ollama](https://github.com/ollama/ollama)本地大语言模型，能够实现生成英文字幕的高准确翻译，自动生成对应的中文字幕。同时也开放了对远程大语言模型的支持，能够实现更快，更准确的批量翻译(batch translation), **强烈推荐使用**。同时批量翻译模式下，具备上下文记忆能力，能够更加准确地翻译句子

## 使用体验：

transcribe(听写)整个流程一般控制在两分钟以内, 30分钟以上的视频可能时间会更久。

翻译耗时：

批量翻译模式 约为视频时长的五分之一左右。
逐句翻译模式 约为视频时长的三分之一左右。
具体效果取决于模型自身。

### 效果：


![image.png](./misc/rag_ibm.png)

![xelite](./misc/xelite.png)

![xelite2](./misc/xelite2.png)


> 部分无法分解的长句效果：


![pc world](./misc/pcworld1.png)

视频效果参考：

[什么是RAG 检索增强生成 【What_is_Retrieval_Augmented_Generation_RAG】](https://www.bilibili.com/video/BV1Kf421d7kj/?vd_source=fc60a3443b9b14ad9f2afef0ca8b093c)


## 配置需求：

一张能够运行CUDA的NVidia 显卡，具体配置需求参考whisper要求

2-10GB 显存的显卡应该都可以使用（2Gb以下的暂未测试过。

whisper模型共有5个大小，体积越大，transcribe精度越高，请根据你的显卡选择合适的模型。

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
| :----: | :--------: | :----------------: | :----------------: | :-----------: | :------------: |
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |

默认使用的是对应的faster whisper large_v2模型。

对于小显存的显卡，也推荐使用huggingface上whisper蒸馏模型，在尽可能的保留large v3 模型的精度的前提下，把显存的使用量从10GB降低到5GB左右。

[**distil-whisper**](https://github.com/huggingface/distil-whisper)

#### 翻译：
项目提供了Ollama模型支持，翻译的准确度，速度依GPU性能 模型性能而定。

对于本地LLM模型，至少需要能够运行14B以上大小，否则完全无法保证翻译的质量。

本地模型建议使用单句翻译模型，批量翻译需要110b以上的模型。


> 作者推荐使用阿里的qwen 1.5 chat系列 模型，qwen 1.5 32b 量化模型在3090上表现良好，能够很好地实现单句准确翻译。


[**qwen 1.5 chat**](https://github.com/langchain-ai/langchain/assets/1011680/f0f0d0c9-f0f0-4f0f-8f0f-8f0f8f0f8f0f)

对于需要高准确度和批量翻译的用户，强烈建议使用大语言模型API，性能远超本地模型。


#### 注意

batch translation 需要大语言模型能够对自己的输出格式有严格的控制能力，**无法控制输出格式的模型尽量不要使用**

batch translation 速度是逐句翻译的3-5倍速度，且更加节省tokens。

> 作者推荐使用阿里的qwen plus模型，qwen plus实现了在翻译质量、api价格、翻译速度上的完美平衡。

[**Qwen plus API**](https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction?spm=a2c4g.11186623.0.0.746b46c1FXZPd1)



## 使用方法

### 1.下载源码

```bash
  git clone https://github.com/hiddenblue/whisperx_Sub.git   # git命令clone源码

  cd whisperx_sub   # cd 进入源码目录
```

首先根据**requirements.txt** 安装依赖

### 2.craete a virtual environment with Python 3.10
`conda create -n whisperx_sub python==3.10`

`conda activate whisperx_sub`

### 3. Install otherr dependency in requirements.txt

*You need install Nvidia drivers for your GPU*

`conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

### 4.配置：

运行前需要在**config.py**文件当中配置运行需要的相关信息

填入目标音频文件的路径，task类型，translation 使用的大语言模型相关的api信息。

#### transcribe 参考配置：

```
# task type
# task could be "transcribe" or "all"

task = "transcribe"

# transcribe parameters

# 需要你配置的一些变量
# 音频或者视频文件路径
audio_file = "./openai_sample.mp3"
```

其中最重要的是配置填入需要处理的音频文件的路径。

task type 工作模式有两种，默认只进行transcribe
1. **transcribe**  只对音频文件进行听写操作。 

2. **all** 则先后对音频文件进行听写，然后利用LLM进行翻译，需要额外配置翻译参数。

#### translation 参考配置：

在配置好transcribe参数的基础上，需配置一个可以调用的大语言模型

config.py 文件当中进行配置

```
# subtitle translation parameters

is_using_local_model = False  # 是否使用本地大语言模型，默认为False

base_url = "http://localhost:11434/api/chat"    # 使用的大语言模型api，本地或者远程

translation_model_name = "qwen:32b-chat-v1.5-q5_K_M"  # LLM模型api

translation_prompt = ""    # 翻译字母时需要用到的prompt提示词，默认可以为空，内置提示词了

srt_file_name = ""   需要单独调用 翻译模式时需要指定的srt文件路径
```

### 运行

直接在terminal或者命令行当中执行whisperx_sub.py文件，或者在IDE中执行

`python whisperx_sub.py`

当看到命令行输出一系列信息后，说明程序开始执行了

根据需要transcribe的视频的时间，等待30s-3min后就能得到听写好的字幕文件

###  输出文件

未经长句分割的字幕文件位于output文件夹下。

output/cut目录下是经过长句分割的字幕文件。

翻译得到的字幕文件为音频文件名字 +CN-ZH.srt文件，位于output目录当中。

```
├── output
│   ├── cut
│   │   ├── openai_sample.json
│   │   ├── openai_sample.srt
│   │   ├── openai_sample.tsv
│   │   ├── openai_sample.txt
│   │   └── openai_sample.vtt
│   ├── openai_sample.json
│   ├── openai_sample.srt
│   ├── openai_sample.tsv
│   ├── openai_sample.txt
│   └── openai_sample.vtt
```


## Todo 

- 更多的语言支持，目前只支持英文（作者只会英语
- 更强的长句断句能力，目前通过手工分析只能解决70%的断句问题
- 图形化界面，需要一个GUI来降低使用门槛，方便广大用户
- 修复 whisperx自身的一些错误，提高transcribe的质量


## 更新日志



## 参考的项目
1. [**Whisper**](https://github.com/openai/whisper)

2. [**Faster-whisper**](https://github.com/SYSTRAN/faster-whisper)

3. [**Whisperx**](https://github.com/m-bain/whisperX)

4. [**deepmultilingualpunctuation**](https://github.com/oliverguhr/deepmultilingualpunctuation)
