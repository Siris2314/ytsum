import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pytube import YouTube


def download_audio_from_youtube(url: str, video_name = "yes") -> str:
    """Downloads audio from a YouTube video.

    Args:
        url (str): YouTube video URL.
        video_name (str): Desired name for the downloaded audio file, default name yes

    Returns:
        str: Path to the downloaded audio file.
    """

    if(len(url) == 0):
        raise ValueError("No Video URL specified")

    video_url = YouTube(url)
    video = video_url.streams.filter(only_audio=True).first()
    name = video_url.title
    filename = f"{name}.mp3"
    video.download(filename=filename)
    return filename, name


def load_speech_recognition_model(model_id="distil-whisper/distil-large-v3") -> pipeline:
    """Loads a speech recognition pipeline model.

    Args:
        model_id (str): Identifier of the model to load. Default "distil-whisper/distil-large-v2"

    Returns:
        pipeline: The loaded speech recognition pipeline.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    whisper = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return whisper