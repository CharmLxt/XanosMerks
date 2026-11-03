"""Приоритеты callback'ов GracefulShutdown.

Меньше число = выполняется раньше. Конвенция:
    1 — сохранить чекпоинт (не потерять прогресс)
    2 — выгрузить кэши, освободить память
    5 — обычные действия (по умолчанию)
    7 — закрыть файлы, соединения
    9 — последнее слово (flush логов, терминальные операции)
"""

PRIORITY_CRITICAL = 1
PRIORITY_HIGH = 2
PRIORITY_NORMAL = 5
PRIORITY_LOW = 7
PRIORITY_LAST = 9


__all__ = [
    "PRIORITY_CRITICAL",
    "PRIORITY_HIGH",
    "PRIORITY_NORMAL",
    "PRIORITY_LOW",
    "PRIORITY_LAST",
]