from typing import List

import whisper


def recognize_audio(
    audio_path: str,
    model_size: str = 'base'
) -> List[dict]:
    """Распознавание речи с использованием модели Whisper.

    Args:
        audio_path: Путь к аудиофайлу для распознавания.
        model_size: Размер модели Whisper (например, 'tiny', 'base', 'small',
                   'medium', 'large'). По умолчанию 'base'.

    Returns:
        Список сегментов распознавания, каждый содержит ключи 'text', 'start'
        и 'end'.
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False)
    return result.get('segments', [])
