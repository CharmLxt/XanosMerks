"""Мягкая интеграция с CharmPkk для XanosMerks.

Если CharmPkk установлен — цветной вывод, баннеры, таблицы.
Если нет — plain print с ASCII-рамками. Всё через один интерфейс.
"""
import sys


_charm = None
_c = None
_available = False

try:
    from charmpkk import console as _c
    _charm = __import__("charmpkk")
    _available = True
except Exception:
    _available = False


def has_charm():
    return _available


def theme(name="halloween"):
    if not _available:
        return None
    try:
        return _c.get_theme(name)
    except Exception:
        return None


def _fg(color):
    if not _available or not color:
        return None
    if isinstance(color, str) and color.startswith("#"):
        try:
            return _c.fg_hex(color)
        except Exception:
            return None
    return color


# ============================================================
# Печать сообщений
# ============================================================

def print_msg(text, color="#7aa2ff"):
    if _available:
        try:
            print(_c.colorize(text, fg=_fg(color)))
            return
        except Exception:
            pass
    print(text)


def colorize(text, color=None, style=None):
    if _available:
        try:
            return _c.colorize(text, fg=_fg(color), style=style)
        except Exception:
            return text
    return text


def banner(title, subtitle=None, color="#7aa2ff"):
    """Крупный заголовок. Возвращает строку."""
    if _available:
        try:
            return _c.banner(title, subtitle, fg=_fg(color))
        except Exception:
            pass
    line = f"\n=== {title} ==="
    if subtitle:
        line += f"\n{subtitle}"
    return line + "\n"


def box(content, title=None, color="#7aa2ff"):
    """Рамка. content — строка или список строк."""
    if _available:
        try:
            return _c.box(content, title=title, border_fg=_fg(color))
        except Exception:
            pass
    lines = content.split("\n") if isinstance(content, str) else [str(x) for x in content]
    max_w = max((len(line) for line in lines), default=0)
    if title:
        max_w = max(max_w, len(title) + 2)
    max_w = min(max_w, 100)
    top = "+" + "-" * (max_w + 2) + "+"
    out = [top]
    if title:
        out.append(f"| {title.center(max_w)} |")
        out.append("+" + "-" * (max_w + 2) + "+")
    for line in lines:
        out.append(f"| {line.ljust(max_w)[:max_w]} |")
    out.append(top)
    return "\n".join(out)


def table(headers, rows, header_color="#7aa2ff"):
    """Простая таблица. Возвращает строку."""
    if _available:
        try:
            return _c.Table(
                headers=headers, rows=rows, header_fg=_fg(header_color),
            ).render()
        except Exception:
            pass
    all_rows = [list(headers)] + [list(r) for r in rows]
    n_cols = len(headers)
    widths = [max(len(str(r[i])) if i < len(r) else 0 for r in all_rows) for i in range(n_cols)]
    lines = []
    for i, row in enumerate(all_rows):
        cells = [str(row[j]) if j < len(row) else "" for j in range(n_cols)]
        lines.append(" | ".join(c.ljust(widths[j]) for j, c in enumerate(cells)))
        if i == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def section(title, color="#7aa2ff"):
    """Горизонтальный разделитель с заголовком."""
    if _available:
        try:
            return _c.rule(width=60, title=title, fg=_fg(color), title_align="left")
        except Exception:
            pass
    return f"\n{'─' * 60}\n  {title}\n{'─' * 60}"


def kv(pairs, key_color="#7aa2ff"):
    """Ключ-значение. pairs — dict или список (k, v)."""
    if _available:
        try:
            data = dict(pairs) if not isinstance(pairs, dict) else pairs
            return _c.key_value(data, key_fg=_fg(key_color))
        except Exception:
            pass
    if isinstance(pairs, dict):
        items = list(pairs.items())
    else:
        items = list(pairs)
    if not items:
        return ""
    max_k = max(len(str(k)) for k, _ in items)
    return "\n".join(f"  {str(k).ljust(max_k)} : {v}" for k, v in items)


def info(msg):
    if _available:
        try:
            print(_c.info(msg))
            return
        except Exception:
            pass
    print(f"[i] {msg}")


def success(msg):
    if _available:
        try:
            print(_c.success(msg))
            return
        except Exception:
            pass
    print(f"[+] {msg}")


def warning(msg):
    if _available:
        try:
            print(_c.warning(msg))
            return
        except Exception:
            pass
    print(f"[!] {msg}")


def error(msg):
    if _available:
        try:
            print(_c.error(msg))
            return
        except Exception:
            pass
    print(f"[x] {msg}")


# ============================================================
# Интерактив
# ============================================================

def ask(prompt, default=None):
    if _available:
        try:
            return _c.ask(prompt, default=default)
        except Exception:
            pass
    hint = f" [{default}]" if default is not None else ""
    try:
        v = input(f"? {prompt}{hint}: ").strip()
        return v or default
    except (EOFError, KeyboardInterrupt):
        return default


def confirm(prompt, default=False):
    if _available:
        try:
            return _c.confirm(prompt, default=default)
        except Exception:
            pass
    d = "Y/n" if default else "y/N"
    try:
        v = input(f"? {prompt} ({d}): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    return v.startswith("y") if v else default


def select(title, options, color="#7aa2ff"):
    if not options:
        return None
    if _available:
        try:
            return _c.select(title, options, fg_cursor=_fg(color), fg_title=_fg(color))
        except Exception:
            pass
    print(title)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    try:
        idx = int(input("> ").strip()) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except Exception:
        return None


# ============================================================
# Прогресс и спиннер
# ============================================================

class _NullProgress:
    def __init__(self, total, label):
        self.total = total
        self.label = label
        self.current = 0
        self._started = False

    def __enter__(self):
        self._started = True
        return self

    def __exit__(self, *exc):
        if sys.stdout.isatty() and self._started:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return False

    def update(self, current):
        self.current = current
        if not sys.stdout.isatty():
            return
        pct = int(current / self.total * 100) if self.total else 0
        bar_w = 30
        filled = int(pct * bar_w / 100)
        bar = "#" * filled + "." * (bar_w - filled)
        sys.stdout.write(f"\r  {self.label} [{bar}] {pct:3d}%")
        sys.stdout.flush()

    def finish(self, msg=None):
        if sys.stdout.isatty() and self._started:
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()
        if msg:
            print(msg)


def progress(total=100, label="", color="#7aa2ff"):
    if _available:
        try:
            return _c.FancyProgressBar(total=total, label=label, fg=_fg(color))
        except Exception:
            pass
    return _NullProgress(total, label)


class _NullSpinner:
    def __init__(self, text):
        self.text = text
        self._printed = False

    def __enter__(self):
        if sys.stdout.isatty():
            sys.stdout.write(f"  {self.text}...")
            sys.stdout.flush()
            self._printed = True
        else:
            print(f"  {self.text}...")
        return self

    def __exit__(self, *exc):
        if self._printed and sys.stdout.isatty():
            sys.stdout.write("\r" + " " * (len(self.text) + 8) + "\r")
            sys.stdout.flush()
        return False

    def stop(self, final=None):
        if final:
            print(f"  {final}")


def spinner(text="Loading", color="#7aa2ff"):
    if _available:
        try:
            return _c.Spinner(text, fg=_fg(color))
        except Exception:
            pass
    return _NullSpinner(text)


# ============================================================
# Кроссплатформенное чтение памяти (тот же _decode_raw из других проектов)
# ============================================================

def decode_raw(raw):
    """utf-8 → cp866 → cp1251. Для консольных утилит Windows."""
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw
    for enc in ("utf-8", "cp866", "cp1251"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, AttributeError):
            continue
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return str(raw)


__all__ = [
    "has_charm", "theme", "colorize",
    "print_msg", "banner", "box", "table", "section", "kv",
    "info", "success", "warning", "error",
    "ask", "confirm", "select",
    "progress", "spinner", "decode_raw",
]