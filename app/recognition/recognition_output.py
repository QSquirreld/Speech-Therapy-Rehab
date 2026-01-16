from typing import Any, Dict, List


def get_recognition_text(segments: List[Dict[str, Any]]) -> str:
    """Форматирует сегменты распознавания в читаемый текст с временными метками.

    Args:
        segments: Список словарей с ключами 'start', 'end' и 'text'.

    Returns:
        Строка с отформатированными сегментами, каждый на новой строке
        в формате "start — end: text".
    """
    lines = []
    for seg in segments:
        start = f"{seg['start']:.2f}"
        end = f"{seg['end']:.2f}"
        lines.append(f"{start} — {end}: {seg['text'].strip()}")
    return '\n'.join(lines)


def extract_plain_text(segments: List[Dict[str, Any]]) -> str:
    """Извлекает только текст из сегментов без временных меток.

    Args:
        segments: Список словарей с ключом 'text'.

    Returns:
        Строка с объединенным текстом всех сегментов через пробел.
    """
    return ' '.join(seg['text'].strip() for seg in segments)
