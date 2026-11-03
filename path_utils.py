"""Утилиты для работы с путями к файлам модели."""
import re
from pathlib import Path

# Регулярка для шардов: model-00001-of-00004.safetensors
SHARD_PATTERN = re.compile(r"^(.+?)-(\d+)-of-(\d+)\.safetensors$")


def list_model_files(folder_path):
    """Возвращает отсортированный список всех файлов в папке (рекурсивно)."""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file()])


def find_shards(folder_path):
    """
    Находит и группирует шарды в папке.
    Возвращает словарь: {base_name: [(index, total, path), ...]}
    """
    folder = Path(folder_path)
    shards = {}
    for f in folder.rglob("*.safetensors"):
        m = SHARD_PATTERN.match(f.name)
        if m:
            base, idx, total = m.group(1), int(m.group(2)), int(m.group(3))
            shards.setdefault(base, []).append((idx, total, f))
    # Сортируем каждый набор по индексу
    for base in shards:
        shards[base].sort(key=lambda x: x[0])
    return shards


def is_shard_complete(shards_info):
    """Проверяет, что все шарды на месте (индексы 1..total)."""
    if not shards_info:
        return False
    indices = [idx for idx, _, _ in shards_info]
    total = shards_info[0][1]
    return indices == list(range(1, total + 1))


def get_shard_total(folder_path):
    """Возвращает общее число шардов или 1, если модель не шардирована."""
    shards = find_shards(folder_path)
    if not shards:
        return 1
    first_base = next(iter(shards))
    return shards[first_base][0][1] if shards[first_base] else 1


def total_size_bytes(paths):
    """Суммарный размер набора файлов в байтах."""
    total = 0
    for p in paths:
        try:
            total += Path(p).stat().st_size
        except OSError:
            pass
    return total


def human_size(num_bytes):
    """Читаемый размер: '4.5 GB'."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def classify_by_extension(files):
    """Группирует файлы по расширениям."""
    result = {}
    for f in files:
        ext = Path(f).suffix.lower()
        result.setdefault(ext, []).append(f)
    return result


def find_config(folder_path):
    """Ищет config.json (учитывая возможные подпапки)."""
    folder = Path(folder_path)
    candidates = list(folder.rglob("config.json"))
    # Фильтруем tokenizer_config.json
    candidates = [c for c in candidates if "tokenizer" not in c.name.lower()]
    return candidates[0] if candidates else None


def find_tokenizer(folder_path):
    """Ищет токенизатор (json/model/tiktoken)."""
    folder = Path(folder_path)
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer.tiktoken", "spiece.model"):
        candidates = list(folder.rglob(name))
        if candidates:
            return candidates[0]
    return None

def find_all_safetensors(folder_path):
    """Находит ВСЕ .safetensors файлы в папке (рекурсивно, включая не-шарды)."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(folder.rglob("*.safetensors"))