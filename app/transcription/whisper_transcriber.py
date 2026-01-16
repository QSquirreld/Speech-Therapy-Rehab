from typing import List

import whisper


def transcribe_audio(
    audio_path: str,
    model_size: str = 'base'
) -> List[dict]:
    """Transcribe audio file using Whisper model.

    Args:
        audio_path: Path to the audio file to transcribe.
        model_size: Whisper model size (e.g., 'tiny', 'base', 'small',
                   'medium', 'large'). Defaults to 'base'.

    Returns:
        List of transcription segments, each containing 'text', 'start',
        and 'end' keys.
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False)
    return result.get('segments', [])
