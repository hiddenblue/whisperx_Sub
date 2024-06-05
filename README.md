# Whisperx_Sub

![GitHub stars](https://img.shields.io/github/stars/hiddenblue/whisperx_Sub?style=social)
![GitHub forks](https://img.shields.io/github/forks/hiddenblue/whisperx_Sub?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/hiddenblue/whisperx_Sub?style=social)
![GitHub repo size](https://img.shields.io/github/repo-size/hiddenblue/whisperx_Sub?)
![GitHub last commit](https://img.shields.io/github/last-commit/hiddenblue/whisperx_Sub?color=red)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhiddenblue%2Fwhisperx_Sub&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

Whisperx_sub is a subtitle generation tool for video transcription and translation, based on Whisperx.

## Language
### [English](README.md)  |   [简体中文](readme/README_zh_CN.md)

## Features:

- Utilizing the [Faster whisper](https://github.com/guillaumekln/faster-whisper) speech model, it can generate English subtitles up to 60x faster than the original Whisper, with a 30-minute video taking only 1-3 minutes to produce complete subtitles.

- The author of [Whisperx](https://github.com/m-bain/whisperX) has addressed the issues of hallucination and inaccurate word pausing in Whisper by leveraging the wav2vec model, achieving precision at the second level.

- By summarizing English sentence segmentation rules and using natural language tools like nltk, it automatically segments longer English sentences without affecting translation, accurately handling 70% of long sentences, significantly reducing the workload of subsequent timing adjustments.

- Utilizing the common [Ollama](https://github.com/ollama/ollama) local large language model, it can achieve highly accurate translation of English subtitles, automatically generating corresponding Chinese subtitles. It also supports remote large language models, enabling faster and more accurate batch translation (batch translation), **strongly recommended for use**. In batch translation mode, it has contextual memory capabilities, allowing for more accurate translation of sentences.

- Currently, the source language for audio or video only supports English (Japanese support will be attempted in the future), while the target translation language supports multiple languages, with translation effectiveness depending on your LLM model.

| Source Language | Target Language |
|:---------------:|:---------------:|
|                 |     Chinese     |
|                 |    Japanese     |
|     English     |     German      |
|                 |     French      |
|                 |       ...       |

## User Experience:

The entire transcribe (transcription) process generally takes under two minutes, with longer videos potentially taking more time.

Translation time:

- Batch translation mode is about one-fifth of the video's duration.
- Sentence-by-sentence translation mode is about one-third of the video's duration.
The actual effectiveness depends on the model itself.

### Results:

<div align=center>
<img src="./misc/rag_ibm.png" alt="rag_ibm.png" width="80%" height="80%">
</div>

<div align=center>
<img src="./misc/xelite.png" alt="xelite.png" width="80%" height="80%">
</div>

<div align=center>
<img src="./misc/xelite2.png" alt="xelite2.png" width="80%" height="80%">
</div>

---

> Results for long sentences that cannot be decomposed:

<div align=center>
<img src="./misc/pcworld1.png" alt="pcworld2.png" width="80%" height="80%">
</div>

**Video Reference:**

[What is RAG Retrieval Augmented Generation [What_is_Retrieval_Augmented_Generation_RAG]](https://www.bilibili.com/video/BV1Kf421d7kj/?vd_source=fc60a3443b9b14ad9f2afef0ca8b093c)

## Configuration Requirements:

A Nvidia graphics card capable of running CUDA, with specific requirements as per Whisper.

Graphics cards with 2-10GB of VRAM should work (cards with less than 2GB have not been tested).

Whisper models come in five sizes, with larger sizes offering higher transcription accuracy. Choose the appropriate model based on your graphics card.

| Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
| :----: | :--------: | :----------------: | :----------------: | :-----------: | :------------: |
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |

The default model used is the corresponding faster whisper large_v2 model.

For graphics cards with smaller VRAM, it is also recommended to use the distilled models from huggingface, which, while retaining as much accuracy as possible from the large v3 model, reduce VRAM usage from 10GB to around 5GB.

[**distil-whisper**](https://github.com/huggingface/distil-whisper)

| Model                                                                      | Params / M | Rel. Latency ↑ | Short-Form WER ↓ | Long-Form WER ↓ |
|----------------------------------------------------------------------------|------------|----------------|------------------|-----------------|
| [large-v3](https://huggingface.co/openai/whisper-large-v3)                 | 1550       | 1.0            | **8.4**          | 11.0            |
| [large-v2](https://huggingface.co/openai/whisper-large-v2)                 | 1550       | 1.0            | 9.1              | 11.7            |
|                                                                            |            |                |                  |                 |
| [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)   | 756        | 6.3            | 9.7              | **10.8**        |
| [distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2)   | 756        | 5.8            | 10.1             | 11.6            |
| [distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en) | 394        | **6.8**        | 11.1             | 12.4            |
| [distil-small.en](https://huggingface.co/distil-whisper/distil-small.en)   | **166**    | 5.6            | 12.1             | 12.8            |

#### Translation:
The project provides support for the Ollama model, with translation accuracy and speed depending on GPU and model performance.

> [!NOTE]
> For local LLM models, a minimum capability of running 14B or larger is required to ensure translation quality.

Local models are recommended for sentence-by-sentence translation, while batch translation requires models larger than 110b.

> Author's recommendation: Use the Alibaba qwen 1.5 chat series model, specifically the qwen 1.5 32b quantized model, which performs well on a 3090 and can accurately translate individual sentences.

[**qwen 1.5 chat**](https://github.com/langchain-ai/langchain/assets/1011680/f0f0d0c9-f0f0-4f0f-8f0f-8f0f8f0f8f0f)

> [!NOTE]
> For users requiring high accuracy and batch translation, it is strongly recommended to use a large language model API, which significantly outperforms local models.

> [!WARNING]
> Batch translation requires the large language model to have strict control over its output format; models that cannot control their output format should be used with caution.
> Batch translation is 3-5 times faster than sentence-by-sentence translation and is more token-efficient.

> Author's recommendation: Use Alibaba's qwen plus model, which achieves a perfect balance in translation quality, API pricing, and translation speed.

[**Qwen plus API**](https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction?spm=a2c4g.11186623.0.0.746b46c1FXZPd1)

## Usage

### 1. Download the Source Code

Clone the source code using git:

```
git clone https://github.com/hiddenblue/whisperx_Sub.git
```

Navigate to the source code directory:

```
cd whisperx_sub
```

Install dependencies based on **requirements.txt**:

### 2. Create a Virtual Environment with Python 3.10

```
conda create -n whisperx_sub python==3.10
```

```
conda activate whisperx_sub
```

### 3. Install Additional Dependencies in requirements.txt

*You need to install Nvidia drivers for your GPU*

```
pip install -r requirements.txt
```

### 4. Configuration:

Before running, configure the necessary information in the **config.py** file:

Enter the path to the target audio file, the task type, and the API information related to the large language model used for translation.

#### Transcribe Configuration Example:

```
# task type
# task could be "transcribe" or "all"

task = "transcribe"

# transcribe parameters

# Variables you need to configure
# Path to the audio or video file
audio_file = "./openai_sample.mp3"
```

The most important part is to configure the path to the audio file that needs to be processed.

There are two task types, with the default being transcribe only:
1. **transcribe** - Only performs transcription on the audio file.
2. **all** - Performs transcription first, then uses the LLM for translation, requiring additional configuration for translation parameters.

#### Translation Configuration Example:

On top of configuring the transcribe parameters, you need to configure a large language model that can be called:

Configure in the config.py file:

```
# subtitle translation parameters

is_using_local_model = False  # Whether to use a local large language model, default is False

base_url = "http://localhost:11434/api/chat"  # API for the large language model, local or remote

translation_model_name = "qwen:32b-chat-v1.5-q5_K_M"  # LLM model API

translation_prompt = ""  # Prompt used for subtitle translation, can be left empty as default prompts are built-in

srt_file_name = ""  # Path to the srt file that needs to be specified when using translation mode
```

### Running

Execute the whisperx_sub.py file directly in the terminal or command line, or run it in an IDE:

```
python whisperx_sub.py
```

Once you see a series of messages output in the command line, the program has started executing.

Depending on the length of the video to be transcribed, wait 30 seconds to 3 minutes to obtain the transcribed subtitle file.

### Output Files

The subtitle files without long sentence segmentation are located in the output folder.

The subtitle files with long sentence segmentation are in the output/cut directory.

The translated subtitle files are named after the audio file with a +CN-ZH.srt extension and are located in the output directory.

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

### Troubleshooting

You may need to install the ffmpeg tool to support various audio and video formats. Installation methods can be found at:
https://github.com/openai/whisper#setup

## Todo

- Support for more languages, currently only English is supported (the author only knows English)
- Improved long sentence segmentation capabilities, currently only 70% of issues can be resolved through manual analysis
- A graphical user interface (GUI) is needed to lower the usage threshold and make it more user-friendly for a wide audience
- Fix some issues with Whisperx itself and improve the quality of transcription

## License 📜

This project is licensed under the [GPL-3.0 license](LICENSE) - see the [LICENSE](LICENSE) file for details. 📄

## Changelog

## References
1. [**Whisper**](https://github.com/openai/whisper)

2. [**Faster-whisper**](https://github.com/SYSTRAN/faster-whisper)

3. [**Whisperx**](https://github.com/m-bain/whisperX)

4. [**deepmultilingualpunctuation**](https://github.com/oliverguhr/deepmultilingualpunctuation)