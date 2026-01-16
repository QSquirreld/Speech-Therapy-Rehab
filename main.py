import argparse
import os

from app.pipeline import analyze_motor_aphasia
from app.transcription.transcript_output import get_transcript_text


def main() -> None:
    """Точка входа для CLI приложения анализа речи.

    Парсит аргументы командной строки и запускает анализ моторной афазии
    для указанного аудиофайла.
    """
    parser = argparse.ArgumentParser(
        description='Analyze speech for motor aphasia.'
    )
    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to audio file (e.g., wav, mp3)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f'File not found: {args.audio_path}')
        return

    print(f'Analyzing: {args.audio_path}')
    result = analyze_motor_aphasia(args.audio_path)

    print('\nSpeech Metrics:')
    for key, value in result.items():
        if key != 'segments':
            print(f'{key}: {value}')

    if result.get('segments'):
        print('\nSpeech Transcription:')
        print(get_transcript_text(result['segments']))


if __name__ == '__main__':
    main()

