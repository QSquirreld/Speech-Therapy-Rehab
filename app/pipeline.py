from typing import Any, Dict, List

from app.analysis.speech_features import (
    compute_onset_latency,
    compute_pause,
    compute_phrase_lengths,
    compute_speech_rate,
)
from app.transcription.whisper_transcriber import transcribe_audio


def analyze_motor_aphasia(audio_path: str) -> Dict[str, Any]:
    """Analyze audio file for motor aphasia indicators.

    Performs transcription and computes speech metrics including
    speech rate, phrase lengths, pauses, and onset latency.

    Args:
        audio_path: Path to the audio file to analyze.

    Returns:
        Dictionary containing speech metrics:
        - speech_rate_wps: Words per second
        - avg_phrase_length: Average words per phrase
        - avg_pause_duration: Average pause duration in seconds
        - pause_count: Total number of pauses detected
        - speech_onset_latency: Time to first speech in seconds
        - total_duration: Total audio duration in seconds
        - segments: List of transcription segments
    """
    segments = transcribe_audio(audio_path)
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
