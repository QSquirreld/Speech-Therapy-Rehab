from typing import List, Tuple

import numpy as np


PAUSE_THRESHOLD_SECONDS = 0.2


def compute_speech_rate(segments: List[dict]) -> float:
    """Compute speech rate in words per second.

    Args:
        segments: List of transcription segments with 'text', 'start',
                  and 'end' keys.

    Returns:
        Speech rate as words per second, or 0.0 if no segments.
    """
    if not segments:
        return 0.0

    total_words = sum(len(seg['text'].split()) for seg in segments)
    total_duration = segments[-1]['end'] - segments[0]['start']
    return total_words / total_duration if total_duration > 0 else 0.0


def compute_phrase_lengths(segments: List[dict]) -> float:
    """Compute average phrase length in words.

    Args:
        segments: List of transcription segments with 'text' key.

    Returns:
        Average words per phrase, or 0.0 if no segments.
    """
    lengths = [len(seg['text'].split()) for seg in segments]
    return float(np.mean(lengths)) if lengths else 0.0


def compute_pause(
    segments: List[dict],
    threshold: float = PAUSE_THRESHOLD_SECONDS
) -> Tuple[float, int]:
    """Compute average pause duration and pause count.

    Args:
        segments: List of transcription segments with 'start' and 'end'
                  keys.
        threshold: Minimum pause duration in seconds to be counted.
                  Defaults to PAUSE_THRESHOLD_SECONDS.

    Returns:
        Tuple of (average_pause_duration, pause_count).
    """
    pauses = []
    for i in range(1, len(segments)):
        pause = segments[i]['start'] - segments[i - 1]['end']
        if pause > threshold:
            pauses.append(pause)

    avg_pause = float(np.mean(pauses)) if pauses else 0.0
    return avg_pause, len(pauses)


def compute_onset_latency(segments: List[dict]) -> float:
    """Compute speech onset latency.

    Args:
        segments: List of transcription segments with 'start' key.

    Returns:
        Time to first speech in seconds, or 0.0 if no segments.
    """
    return segments[0]['start'] if segments else 0.0
