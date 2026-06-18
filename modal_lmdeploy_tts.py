import os
import subprocess

import modal


APP_NAME = "vieneutts-lmdeploy"
MODEL_ID = os.environ.get("LMDEPLOY_MODEL", "pnnbao-ump/VieNeu-TTS")
MODEL_NAME = os.environ.get("LMDEPLOY_MODEL_NAME", MODEL_ID)
PORT = int(os.environ.get("LMDEPLOY_PORT", "23333"))


app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("vieneutts-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("git", "curl")
    .pip_install("lmdeploy")
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
            "LMDEPLOY_MODEL": MODEL_ID,
            "LMDEPLOY_MODEL_NAME": MODEL_NAME,
            "LMDEPLOY_PORT": str(PORT),
            "LMDEPLOY_TP": os.environ.get("LMDEPLOY_TP", "1"),
            "LMDEPLOY_CACHE_MAX_ENTRY_COUNT": os.environ.get(
                "LMDEPLOY_CACHE_MAX_ENTRY_COUNT",
                "0.3",
            ),
        }
    )
)


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "A10G"),
    timeout=60 * 60,
    startup_timeout=20 * 60,
    scaledown_window=10 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.web_server(PORT, startup_timeout=20 * 60)
def lmdeploy_server():
    command = [
        "lmdeploy",
        "serve",
        "api_server",
        os.environ.get("LMDEPLOY_MODEL", MODEL_ID),
        "--server-name",
        "0.0.0.0",
        "--server-port",
        str(PORT),
        "--tp",
        os.environ.get("LMDEPLOY_TP", "1"),
        "--cache-max-entry-count",
        os.environ.get("LMDEPLOY_CACHE_MAX_ENTRY_COUNT", "0.3"),
        "--model-name",
        os.environ.get("LMDEPLOY_MODEL_NAME", MODEL_NAME),
    ]

    subprocess.Popen(command)
