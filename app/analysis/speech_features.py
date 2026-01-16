from typing import List, Tuple

import numpy as np


PAUSE_THRESHOLD_SECONDS = 0.2


def compute_speech_rate(segments: List[dict]) -> float:
    """Вычисляет скорость речи в словах в секунду.

    Args:
        segments: Список сегментов транскрипции с ключами 'text', 'start'
                  и 'end'.

    Returns:
        Скорость речи в словах в секунду, или 0.0 если сегментов нет.
    """
    if not segments:
        return 0.0

    total_words = sum(len(seg['text'].split()) for seg in segments)
    total_duration = segments[-1]['end'] - segments[0]['start']
    return total_words / total_duration if total_duration > 0 else 0.0


def compute_phrase_lengths(segments: List[dict]) -> float:
    """Вычисляет среднюю длину фразы в словах.

    Args:
        segments: Список сегментов транскрипции с ключом 'text'.

    Returns:
        Среднее количество слов во фразе, или 0.0 если сегментов нет.
    """
    lengths = [len(seg['text'].split()) for seg in segments]
    return float(np.mean(lengths)) if lengths else 0.0


def compute_pause(
    segments: List[dict],
    threshold: float = PAUSE_THRESHOLD_SECONDS
) -> Tuple[float, int]:
    """Вычисляет среднюю длительность паузы и количество пауз.

    Args:
        segments: Список сегментов транскрипции с ключами 'start' и 'end'.
        threshold: Минимальная длительность паузы в секундах для подсчета.
                  По умолчанию PAUSE_THRESHOLD_SECONDS.

    Returns:
        Кортеж из (average_pause_duration, pause_count).
    """
    pauses = []
    for i in range(1, len(segments)):
        pause = segments[i]['start'] - segments[i - 1]['end']
        if pause > threshold:
            pauses.append(pause)

    avg_pause = float(np.mean(pauses)) if pauses else 0.0
    return avg_pause, len(pauses)


def compute_onset_latency(segments: List[dict]) -> float:
    """Вычисляет латентность начала речи.

    Args:
        segments: Список сегментов транскрипции с ключом 'start'.

    Returns:
        Время до начала речи в секундах, или 0.0 если сегментов нет.
    """
    return segments[0]['start'] if segments else 0.0
