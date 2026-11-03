"""CLI-движок meanflare: загрузка, конвертация, lazy-индекс, стресс-тест."""
import sys
import json
import time
import argparse
from pathlib import Path

from .loaders import (
    load_safetensors, load_gguf, load_npy, load_model_weights,
    _read_safetensors_header,
)
from .convert import (
    convert_model, describe_conversion, compare_dtype_sizes,
    detect_optimal_dtype,
)
from .memory import (
    MemoryManager, TensorHandle, TensorIndex,
    create_handles_from_safetensors,
)
from .support_tools.core_spr.exceptions import MeanflareError
from .. import charm_enhance as ui


def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _section(title):
    if ui.has_charm():
        print(ui.section(title))
    else:
        print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _print_kv(pairs, indent=2):
    if not pairs:
        return
    if ui.has_charm():
        try:
            print(ui.kv(dict(pairs) if not isinstance(pairs, dict) else pairs))
            return
        except Exception:
            pass
    pad = " " * indent
    max_key = max(len(str(k)) for k, _ in pairs)
    for k, v in pairs:
        print(f"{pad}{str(k).ljust(max_key)} : {v}")


def _detect_format(path):
    path = Path(path)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".safetensors": return "safetensors", [path]
        if suffix == ".gguf": return "gguf", [path]
        if suffix in (".npy", ".npz"): return "npy", [path]
        return "unknown", [path]

    st_files = sorted(path.rglob("*.safetensors"))
    if st_files: return "safetensors", st_files
    gguf_files = sorted(path.rglob("*.gguf"))
    if gguf_files: return "gguf", gguf_files
    npy_files = sorted(path.rglob("*.npy")) + sorted(path.rglob("*.npz"))
    if npy_files: return "npy", npy_files
    return "unknown", []


def cmd_inspect(args):
    path = Path(args.path)
    if not path.exists():
        print(f"[ERROR] Путь не существует: {path}", file=sys.stderr)
        return 1

    fmt, files = _detect_format(path)
    if not files:
        print(f"[ERROR] Не найдено ни .safetensors, ни .gguf, ни .npy", file=sys.stderr)
        return 1

    total_size = sum(f.stat().st_size for f in files)

    _section(f"Model Info ({fmt})")
    _print_kv([
        ("Path", str(path)),
        ("Format", fmt),
        ("Files", len(files)),
        ("Total size", _human(total_size)),
    ])

    if fmt == "safetensors":
        total_tensors = 0
        for f in files[:10]:
            try:
                header, header_size = _read_safetensors_header(f)
                n_tensors = len([k for k in header if k != "__metadata__"])
                total_tensors += n_tensors
                print(f"    {f.name}: {n_tensors} tensors, header={_human(header_size)}")
            except Exception as e:
                print(f"    {f.name}: [ERROR] {e}")
        if len(files) > 10:
            print(f"    ... и ещё {len(files) - 10} файлов")
        print(f"\n    Всего тензоров: {total_tensors}")

    elif fmt == "gguf":
        from ..gamtype.readers import parse_gguf
        for f in files:
            try:
                info = parse_gguf(f)
                print(f"    {f.name}:")
                print(f"      version: {info.get('version')}")
                print(f"      architecture: {info.get('architecture', '?')}")
                print(f"      tensors: {info.get('tensor_count')}")
                print(f"      metadata: {info.get('metadata_count')} keys")
            except Exception as e:
                print(f"    {f.name}: [ERROR] {e}")

    return 0


def cmd_load(args):
    path = Path(args.path)
    if not path.exists():
        print(f"[ERROR] Путь не существует: {path}", file=sys.stderr)
        return 1

    t0 = time.time()
    try:
        tensors, meta = load_model_weights(
            path, dtype=args.dtype, mmap=not args.no_mmap,
        )
    except MeanflareError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0

    total_bytes = sum(a.nbytes for a in tensors.values())

    _section(f"Loaded model ({meta['format']})")
    _print_kv([
        ("Tensors", meta["num_tensors"]),
        ("Total bytes", _human(total_bytes)),
        ("Load time", f"{elapsed*1000:.0f} ms"),
        ("dtype", args.dtype),
        ("mmap", not args.no_mmap),
    ])

    if args.show_tensors:
        print("\n  Tensors:")
        for name in sorted(tensors.keys())[:args.show_tensors]:
            arr = tensors[name]
            shape_str = "x".join(str(d) for d in arr.shape)
            print(f"    {name:<50} [{shape_str}] {arr.dtype} {_human(arr.nbytes)}")
        if len(tensors) > args.show_tensors:
            print(f"    ... и ещё {len(tensors) - args.show_tensors} тензоров")

    return 0


def cmd_convert(args):
    path = Path(args.path)
    if not path.exists():
        print(f"[ERROR] Путь не существует: {path}", file=sys.stderr)
        return 1

    try:
        tensors, meta = load_model_weights(path, dtype="auto", mmap=True)
    except MeanflareError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    source_dtypes = meta.get("source_dtypes")
    target = args.to
    if target == "auto":
        target = detect_optimal_dtype(tensors, source_dtype_map=source_dtypes)

    plan = describe_conversion(tensors, target)

    _section(f"Conversion Plan (→ {target})")
    _print_kv([
        ("Before", plan["before_human"]),
        ("After", plan["after_human"]),
        ("Ratio", f"{plan['ratio']:.2f}x"),
        ("Savings", f"{plan['savings_pct']}%"),
    ])

    if args.apply:
        print("\n  Применяю конвертацию...")
        t0 = time.time()
        converted, stats = convert_model(
            tensors, target, source_dtype_map=source_dtypes,
        )
        elapsed = time.time() - t0
        _section("Conversion Result")
        _print_kv([
            ("Converted tensors", stats["converted"]),
            ("Skipped", stats["skipped"]),
            ("Time", f"{elapsed*1000:.0f} ms"),
        ])
        first = list(converted.keys())[0]
        print(f"\n  Пример: {first} → {converted[first].dtype}")

    return 0


def cmd_dtype_compare(args):
    path = Path(args.path)
    try:
        tensors, meta = load_model_weights(path, dtype="auto", mmap=True)
    except MeanflareError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    sizes = compare_dtype_sizes(tensors)
    optimal = detect_optimal_dtype(tensors, source_dtype_map=meta.get("source_dtypes"))

    _section("Size Comparison by dtype")
    for dt in ("float32", "float16", "bfloat16", "int8", "fp8_e4m3"):
        if dt in sizes:
            mark = " ← optimal" if dt == optimal else ""
            print(f"    {dt:<12} {_human(sizes[dt]):>12}{mark}")

    return 0


def cmd_lazy(args):
    path = Path(args.path)
    if not path.exists():
        print(f"[ERROR] Путь не существует: {path}", file=sys.stderr)
        return 1

    fmt, files = _detect_format(path)
    if fmt != "safetensors":
        print(f"[ERROR] Lazy режим поддерживает только .safetensors (сейчас: {fmt})", file=sys.stderr)
        return 1

    mgr = MemoryManager(
        max_bytes=(args.budget_mb * 1024 * 1024) if args.budget_mb else None,
        eviction_policy=args.eviction,
    )

    t0 = time.time()
    try:
        idx = create_handles_from_safetensors(
            path, memory_manager=mgr, target_dtype=args.target_dtype,
        )
    except MeanflareError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0

    _section("Lazy Index")
    s = idx.summary()
    _print_kv([
        ("Handles", len(idx)),
        ("Format", "safetensors"),
        ("Budget", _human(mgr.max_bytes) if mgr.max_bytes else "unlimited"),
        ("Disk total", s["total_disk_human"]),
        ("RAM estimate", s["total_ram_human"]),
        ("Index build time", f"{elapsed*1000:.0f} ms"),
    ])
    print(f"\n  Index: {idx}")
    print(f"  Manager: {mgr}")

    if args.load_pattern:
        _section(f"Loading pattern: {args.load_pattern}")
        try:
            matches = idx[args.load_pattern]
            if isinstance(matches, dict):
                print(f"    Найдено {len(matches)} тензоров по паттерну")
                for name in list(matches.keys())[:5]:
                    arr = matches[name].load()
                    print(f"    OK {name}: {arr.shape} {arr.dtype}")
            else:
                arr = matches.load()
                print(f"    OK {args.load_pattern}: {arr.shape} {arr.dtype}")
        except KeyError as e:
            print(f"    [ERROR] {e}")
        print(f"\n  After load: {mgr}")

    return 0


def cmd_stress(args):
    path = Path(args.path)
    fmt, files = _detect_format(path)
    if fmt != "safetensors":
        print(f"[ERROR] Стресс-тест только для .safetensors", file=sys.stderr)
        return 1

    budget_bytes = (args.budget_mb or 1) * 1024 * 1024
    mgr = MemoryManager(
        max_bytes=budget_bytes,
        eviction_policy="lru",
    )
    idx = create_handles_from_safetensors(path, memory_manager=mgr)

    _section(f"Stress test (budget = {_human(budget_bytes)}, policy = lru)")
    print(f"  Загружаем {len(idx)} тензоров последовательно...")

    evictions = 0
    peak_used = 0
    try:
        for i, name in enumerate(idx.keys(), 1):
            idx.load(name)
            peak_used = max(peak_used, mgr.used_bytes)
            if i % max(1, len(idx) // 10) == 0:
                print(f"    [{i}/{len(idx)}] used={_human(mgr.used_bytes)}, peak={_human(mgr.stats.peak_usage)}")
    except MeanflareError as e:
        print(f"  [ERROR] {e}")
        return 1
    except MemoryError as e:
        print(f"  [ERROR] {e}")
        return 1

    s = mgr.summary()
    _section("Result")
    _print_kv([
        ("All tensors processed", len(idx)),
        ("Peak usage", s["peak_human"]),
        ("Budget", _human(budget_bytes)),
        ("Currently loaded", s["loaded"]),
        ("Used after stress", s["used_human"]),
    ])
    if peak_used <= budget_bytes:
        print(f"\n  OK Бюджет НЕ превышен (пик {_human(peak_used)} <= {_human(budget_bytes)})")
        return 0
    else:
        print(f"\n  FAIL Бюджет ПРЕВЫШЕН: пик {_human(peak_used)} > {_human(budget_bytes)}")
        return 2


def build_parser():
    parser = argparse.ArgumentParser(
        prog="xanosmerks-meanflare",
        description="meanflare — загрузчик весов моделей (safetensors, gguf, npy).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="Метаданные файла весов (без загрузки в RAM)")
    p.add_argument("path")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("load", help="Загрузить модель в RAM")
    p.add_argument("path")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "float32", "bfloat16"])
    p.add_argument("--no-mmap", action="store_true", help="Без mmap (полная загрузка)")
    p.add_argument("--show-tensors", type=int, default=0, metavar="N")
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("convert", help="План конвертации dtype")
    p.add_argument("path")
    p.add_argument("--to", default="float16",
                   choices=["auto", "float16", "float32", "bfloat16"])
    p.add_argument("--apply", action="store_true", help="Применить конвертацию")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("dtype-compare", help="Сравнить размеры во всех dtype")
    p.add_argument("path")
    p.set_defaults(func=cmd_dtype_compare)

    p = sub.add_parser("lazy", help="Lazy-индекс (без загрузки в RAM)")
    p.add_argument("path")
    p.add_argument("--budget-mb", type=float, default=None)
    p.add_argument("--eviction", default="lru", choices=["lru", "none"])
    p.add_argument("--load-pattern", default=None)
    p.add_argument("--target-dtype", default="auto",
                   choices=["auto", "float16", "float32", "bfloat16"])
    p.set_defaults(func=cmd_lazy)

    p = sub.add_parser("stress", help="Стресс-тест LRU eviction")
    p.add_argument("path")
    p.add_argument("--budget-mb", type=float, default=1.0)
    p.set_defaults(func=cmd_stress)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())