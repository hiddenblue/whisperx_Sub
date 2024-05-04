import whisperx
import gc
import torch

import copy
import numpy as np

def transcribe(audio_file:str,  # auduio file
               device="cuda",
               batch_size=16,
               compute_type="float16",
               model_dir="./",
               output_dir="./") -> dict:

    # some VAD args
    vad_onset = 0.500
    vad_offset = 0.363
    chunk_size = 30

    # some asr options, we use whisperx default cli args
    temperature_increment_on_fallback = 0.2
    asr_options = {
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "temperatures": 0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "suppress_tokens": "-1",
        "suppress_numerals": False
    }

    temperature = asr_options["temperatures"]
    if (increment := temperature_increment_on_fallback) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 8
    if faster_whisper_threads > 0:
        torch.set_num_threads(faster_whisper_threads)

    asr_options["suppress_tokens"] = [int(x) for x in "-1".split(",")]

    # 1. Transcribe with original whisper (batched)
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root="./")

    # save model to local path (optional)
    gc.collect()
    torch.cuda.empty_cache()

    model = whisperx.load_model("large-v2",
                                device,
                                compute_type=compute_type,
                                download_root=model_dir,
                                asr_options=asr_options,
                                vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
                                threads=faster_whisper_threads)


    result = model.transcribe(audio_file, batch_size=batch_size, chunk_size= chunk_size, print_progress=True)
    print(result["segments"])  # before alignment
    print(len(result["segments"]))

    # 保留一个transribe的结果，方便调试
    transcribe_result = copy.deepcopy(result)

    # delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model

    print("**************" * 20)
    # 2. Align whisper output
    # you can specify model_name here
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

    print(">>Performing alignment...")
    result = whisperx.align(result["segments"],
                            model_a,
                            metadata,
                            audio_file,
                            device,
                            return_char_alignments=False,
                            print_progress=True)

    print(result["segments"])  # after alignment
    print(len(result["segments"]))

    # delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


