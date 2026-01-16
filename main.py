import argparse
import os
from pprint import pprint

from app.pipeline import analyze_motor_aphasia


def main() -> None:
    """Entry point for CLI speech analysis application.

    Parses command-line arguments and runs motor aphasia analysis
    on the provided audio file.
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


if __name__ == '__main__':
    main()
