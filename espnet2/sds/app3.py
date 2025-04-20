import argparse
import os
import shutil
import time
from typing import Generator, Optional, Tuple

import gradio as gr
import nltk
import numpy as np
import torch
from huggingface_hub import HfApi

from espnet_model import ESPnetSDSModelInterface
from dotenv import load_dotenv
# ------------------------
# Hyperparameters
# ------------------------


load_dotenv()

# ---------------------------------------------------------------- helpers
def getenv(name: str) -> str:
    val = os.getenv(name)
    if not val:
        sys.exit(f"✖  Set the {name} environment variable.")
    return val


access_token = os.getenv("HF_TOKEN")

# print(access_token)
ASR_name = None
LLM_name = None
TTS_name = None
ASR_options = []
LLM_options = []
TTS_options = []
upload_to_hub = None
dialogue_model = None

latency_ASR = 0.0
latency_LM = 0.0
latency_TTS = 0.0

text_str = ""
asr_output_str = ""
vad_output = None
audio_output = None
audio_output1 = None
flag = True
LLM_response_arr = []
total_response_arr = []
callback = gr.CSVLogger()
start_record_time = None
enable_btn = gr.Button(interactive=True, visible=True)

# ------------------------
# Function Definitions
# ------------------------


def parse_args():
    global access_token
    global ASR_name
    global LLM_name
    global TTS_name
    global ASR_options
    global LLM_options
    global TTS_options
    global upload_to_hub
    global dialogue_model
    parser = argparse.ArgumentParser(description="Run the app.")
    parser.add_argument(
        "--asr_options",
        #required=True,
        default="whisper-large",
        help="Provide the possible ASR options available to user.",
    )
    parser.add_argument(
        "--llm_options",
        #required=True,
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="Provide the possible LLM options available to user.",
    )
    parser.add_argument(
        "--tts_options",
        #required=True,
        default="kan-bayashi/ljspeech_vits",
        help="Provide the possible TTS options available to user.",
    )
    parser.add_argument(
        "--default_asr_model",
        required=False,
        default="whisper-large",
        help="Provide the default ASR model.",
    )
    parser.add_argument(
        "--default_llm_model",
        required=False,
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="Provide the default LLM model.",
    )
    parser.add_argument(
        "--default_tts_model",
        required=False,
        default="kan-bayashi/ljspeech_vits",
        help="Provide the default TTS model.",
    )
    parser.add_argument(
        "--upload_to_hub",
        required=False,
        default=None,
        help="Hugging Face dataset to upload user data",
    )
    args = parser.parse_args()
    ASR_name = args.default_asr_model
    LLM_name = args.default_llm_model
    TTS_name = args.default_tts_model
    ASR_options = args.asr_options
    LLM_options = args.llm_options
    TTS_options = args.tts_options
    upload_to_hub = args.upload_to_hub
    print(access_token)
    dialogue_model = ESPnetSDSModelInterface(
        ASR_name, LLM_name, TTS_name, "Cascaded", access_token
    )

def start_warmup():
    """
    Initializes and warms up the dialogue and evaluation model.

    This function is designed to ensure that all
    components of the dialogue model are pre-loaded
    and ready for execution, avoiding delays during runtime.
    """
    global dialogue_model
    global ASR_options
    global LLM_options
    global TTS_options
    global ASR_name
    global LLM_name
    global TTS_name
    opt_count = len(ASR_options)
    print("Loading ASR model: " + ASR_options[:opt_count])
    opt = ASR_options[:opt_count]
    try:
        for _ in dialogue_model.handle_ASR_selection(opt):
            continue
    except Exception:
        print("Removing " + opt + " from ASR options since it cannot be loaded.")
        ASR_options = ASR_options[:opt_count] + ASR_options[(opt_count + 1) :]
        if opt == ASR_name:
            ASR_name = ASR_options[0]

    opt_count = len(LLM_options)
    print("Loading LLM model: " + LLM_options[:opt_count])
    opt = LLM_options[:opt_count]
    try:
        for _ in dialogue_model.handle_LLM_selection(opt):
            continue
    except Exception as e:
        print(f"[ERROR] Failed to load {opt}: {str(e)}")
        print("Removing " + opt + " from LLM options since it cannot be loaded.")
        LLM_options = LLM_options[:opt_count] + LLM_options[(opt_count + 1) :]
        if opt == LLM_name:
            LLM_name = LLM_options[0]
            
    opt_count = len(TTS_options)
    print("Loading TTS model: " + TTS_options[:opt_count])
    opt = TTS_options[:opt_count]
    try:
        for _ in dialogue_model.handle_TTS_selection(opt):
            continue
    except Exception as e:
        print(f"[ERROR] Failed to load {opt}: {str(e)}")
        print("Removing " + opt + " from TTS options since it cannot be loaded.")
        TTS_options = TTS_options[:opt_count] + TTS_options[(opt_count + 1) :]
        if opt == TTS_name:
            TTS_name = TTS_options[0]
        raise Exception

    #dialogue_model.handle_E2E_selection()
    dialogue_model.client = None
    for _ in dialogue_model.handle_TTS_selection(TTS_name):
        continue
    for _ in dialogue_model.handle_ASR_selection(ASR_name):
        continue
    for _ in dialogue_model.handle_LLM_selection(LLM_name):
        continue


def flash_buttons():
    """
    Enables human feedback buttons after displaying system output.
    """
    btn_updates = (enable_btn,) * 8
    yield (
        "",
        "",
    ) + btn_updates


def transcribe_full_audio(
    audio: Tuple[int, np.ndarray],
    TTS_option: str,
    ASR_option: str,
    LLM_option: str,
    type_option: str,
):
    global text_str, audio_output, audio_output1
    global asr_output_str, latency_ASR, latency_LM, latency_TTS
    global LLM_response_arr, total_response_arr
    print(audio)
    sr, y = audio

    # Call the dialogue model once for the full clip
    (
        asr_output_str,
        text_str,
        audio_output,
        audio_output1,
        latency_ASR,
        latency_LM,
        latency_TTS,
        _,  # skip updated stream
        change,
    ) = dialogue_model(
        y,
        sr,
        y,  # pass full clip as "stream"
        "", "", None, None,
        0.0, 0.0, 0.0
    )

    if change:
        if asr_output_str:
            total_response_arr.append(asr_output_str.replace("\n", " "))
        LLM_response_arr.append(text_str.replace("\n", " "))
        total_response_arr.append(text_str.replace("\n", " "))

    return asr_output_str, text_str, audio_output, audio_output1





# ------------------------
# Executable Script
# ------------------------

parse_args()
api = HfApi()
start_warmup()

with gr.Blocks(title="E2E Spoken Dialog System") as demo:
    with gr.Row():
        gr.Markdown("""
            Welcome to the E2E Spoken Dialog System demo!
            This demo showcases a conversational AI system that can
            understand and respond to spoken language.
            You can interact with the system by recording your voice.
        """)

    with gr.Row():
        with gr.Column(scale=1):
            user_audio = gr.Audio(
                sources=["microphone"],
                type="numpy",  # ensures we get (sr, waveform)
                label="Record your voice"
            )
            with gr.Row():
                type_radio = gr.Radio(choices=["Cascaded"], label="Type", value="Cascaded")
            with gr.Row():
                ASR_radio = gr.Radio(choices=[ASR_options], label="ASR", value=ASR_name)
            with gr.Row():
                LLM_radio = gr.Radio(choices=[LLM_options], label="LLM", value=LLM_name)
            with gr.Row():
                radio = gr.Radio(choices=[TTS_options], label="TTS", value=TTS_name)

        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Output", autoplay=True)
            output_audio1 = gr.Audio(label="Output1", autoplay=False, visible=False)
            output_asr_text = gr.Textbox(label="ASR output")
            output_text = gr.Textbox(label="LLM output")

    # Bind submit
    user_audio.change(
        fn=transcribe_full_audio,
        inputs=[user_audio, radio, ASR_radio, LLM_radio, type_radio],
        outputs=[output_asr_text, output_text, output_audio, output_audio1],
    )
demo.launch(share=True)
