"""Низкоуровневое чтение бинарных данных (u8/u32/f32/строки)."""
import struct
from pathlib import Path


class ByteReader:
    """Обёртка над бинарным файлом с курсором и удобными методами чтения."""
    def __init__(self, filepath):
        self.path = Path(filepath)
        self.f = open(self.path, "rb")
        self.cursor = 0

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def seek(self, offset):
        self.f.seek(offset)
        self.cursor = offset

    def tell(self):
        return self.cursor

    def read(self, n):
        data = self.f.read(n)
        if len(data) != n:
            raise EOFError(f"Ожидалось {n} байт, получено {len(data)} (файл: {self.path.name})")
        self.cursor += n
        return data

    # --- Целые числа (little-endian, как в safetensors/gguf) ---
    def u8(self):  return struct.unpack("<B", self.read(1))[0]
    def i8(self):  return struct.unpack("<b", self.read(1))[0]
    def u16(self): return struct.unpack("<H", self.read(2))[0]
    def i16(self): return struct.unpack("<h", self.read(2))[0]
    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def i32(self): return struct.unpack("<i", self.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.read(8))[0]
    def i64(self): return struct.unpack("<q", self.read(8))[0]

    # --- Числа с плавающей точкой ---
    def f32(self): return struct.unpack("<f", self.read(4))[0]
    def f64(self): return struct.unpack("<d", self.read(8))[0]

    # --- Строки в разных форматах ---

    def read_gguf_string(self):
        """GGUF-строка: u64 длина, затем UTF-8 байты."""
        length = self.u64()
        return self.read(length).decode("utf-8", errors="replace")

    def read_hf_string(self):
        """HuggingFace-строка: u64 длина, затем UTF-8."""
        # Совпадает с GGUF, но семантически другой формат
        length = self.u64()
        return self.read(length).decode("utf-8", errors="replace")

    def read_pascal_string(self):
        """Pascal-строка: u8 длина, затем байты (для .bin-форматов)."""
        length = self.u8()
        return self.read(length).decode("utf-8", errors="replace")

    def read_cstring(self, encoding="utf-8"):
        """Null-terminated строка (C-стиль)."""
        buf = bytearray()
        while True:
            b = self.read(1)[0]
            if b == 0:
                break
            buf.append(b)
        return bytes(buf).decode(encoding, errors="replace")

    def read_until(self, separator):
        """Читает байты до разделителя (не включая его)."""
        if isinstance(separator, str):
            separator = separator.encode()
        buf = bytearray()
        while True:
            b = self.read(1)
            if b == separator[:1]:
                # Проверяем, что дальше идёт остальная часть разделителя
                rest = self.read(len(separator) - 1)
                if separator[:1] + rest == separator:
                    break
                buf.extend(b + rest)
            else:
                buf.extend(b)
        return bytes(buf)

    def read_bytes(self, n):
        """Читает n сырых байтов."""
        return self.read(n)


# ============================================================
# Утилиты без класса (быстрое чтение заголовков)
# ============================================================

def peek_magic(filepath, n_bytes=8):
    """Возвращает первые n байт файла (для определения формата)."""
    with open(filepath, "rb") as f:
        return f.read(n_bytes)


def file_size(filepath):
    """Размер файла в байтах."""
    return Path(filepath).stat().st_size


def is_safetensors(filepath):
    """Проверка по magic bytes: первые 8 байт = u64 размер заголовка, потом '{'."""
    try:
        with open(filepath, "rb") as f:
            size_bytes = f.read(8)
            if len(size_bytes) != 8:
                return False
            size = struct.unpack("<Q", size_bytes)[0]
            header_start = f.read(1)
            return header_start == b"{"
    except Exception:
        return False


def is_gguf(filepath):
    """Проверка по magic bytes: 'GGUF'."""
    try:
        with open(filepath, "rb") as f:
            return f.read(4) == b"GGUF"
    except Exception:
        return False


def is_npy(filepath):
    """Проверка по magic bytes: \\x93NUMPY."""
    try:
        with open(filepath, "rb") as f:
            return f.read(6) == b"\x93NUMPY"
    except Exception:
        return False