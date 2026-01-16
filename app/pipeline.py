from typing import Any, Dict, List

from app.analysis.speech_features import (
    compute_onset_latency,
    compute_pause,
    compute_phrase_lengths,
    compute_speech_rate,
)
from app.recognition.whisper_recognizer import recognize_audio


def analyze_motor_aphasia(audio_path: str) -> Dict[str, Any]:
    """Анализирует аудиофайл на признаки моторной афазии.

    Выполняет распознавание и вычисляет метрики речи, включая
    скорость речи, длину фраз, паузы и латентность начала речи.

    Args:
        audio_path: Путь к аудиофайлу для анализа.

    Returns:
        Словарь, содержащий метрики речи:
        - speech_rate_wps: Слов в секунду
        - avg_phrase_length: Среднее количество слов во фразе
        - avg_pause_duration: Средняя длительность паузы в секундах
        - pause_count: Общее количество обнаруженных пауз
        - speech_onset_latency: Время до начала речи в секундах
        - total_duration: Общая длительность аудио в секундах
        - segments: Список сегментов распознавания
    """
    segments = recognize_audio(audio_path)
    speech_rate = compute_speech_rate(segments)
    avg_phrase_len = compute_phrase_lengths(segments)
    avg_pause, pause_count = compute_pause(segments)
    onset_latency = compute_onset_latency(segments)

    total_duration = 0.0
    if segments:
        total_duration = round(
            segments[-1]['end'] - segments[0]['start'], 2
        )

    return {
        'speech_rate_wps': round(speech_rate, 2),
        'avg_phrase_length': round(avg_phrase_len, 2),
        'avg_pause_duration': round(avg_pause, 2),
        'pause_count': pause_count,
        'speech_onset_latency': round(onset_latency, 2),
        'total_duration': total_duration,
        'segments': segments,
    }
