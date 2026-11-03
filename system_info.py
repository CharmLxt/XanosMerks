"""Кроссплатформенное чтение системной памяти без внешних зависимостей.

Все функции возвращают байты (int) или None. Никаких исключений наружу.
"""
import platform

_SYSTEM = platform.system()


def _win_system_memory():
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        return None
    return {
        "total": int(mem.ullTotalPhys),
        "available": int(mem.ullAvailPhys),
        "used_percent": float(mem.dwMemoryLoad),
    }


def _win_process_counters():
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = kernel32.GetCurrentProcess()

    func = None
    try:
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        func = kernel32.K32GetProcessMemoryInfo
    except AttributeError:
        pass

    if func is None:
        try:
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            func = psapi.GetProcessMemoryInfo
        except (OSError, AttributeError):
            return None, None

    ok = func(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return None, None
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _linux_system_memory():
    info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                k, _, v = line.partition(":")
                parts = v.strip().split()
                if not parts:
                    continue
                try:
                    n = int(parts[0]) * 1024
                except ValueError:
                    continue
                if k == "MemTotal":
                    info["total"] = n
                elif k == "MemAvailable":
                    info["available"] = n
    except OSError:
        return None
    if "total" in info and "available" in info:
        used = info["total"] - info["available"]
        info["used_percent"] = (used / info["total"]) * 100.0
    return info or None


def _linux_process_status():
    rss = peak = None
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) * 1024
    except OSError:
        return None, None
    return rss, peak


def _macos_system_memory():
    try:
        import subprocess
        total = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip())
    except Exception:
        return None
    available = None
    try:
        import subprocess
        out = subprocess.check_output(["vm_stat"], text=True)
        page_size = 4096
        free_pages = 0
        for line in out.splitlines():
            if "page size of" in line:
                parts = line.split("page size of")
                page_size = int(parts[1].strip().split()[0])
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":")[1].strip().rstrip("."))
        if free_pages > 0:
            available = free_pages * page_size
    except Exception:
        available = None
    result = {"total": total, "available": available}
    if available is not None and total > 0:
        result["used_percent"] = ((total - available) / total) * 100.0
    return result


def _macos_process_rss():
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return None, int(usage.ru_maxrss)
    except Exception:
        return None, None


def get_system_memory():
    """Возвращает dict с системной памятью или None."""
    try:
        if _SYSTEM == "Windows":
            return _win_system_memory()
        if _SYSTEM == "Linux":
            return _linux_system_memory()
        if _SYSTEM == "Darwin":
            return _macos_system_memory()
    except Exception:
        return None
    return None


def get_process_rss():
    """RSS текущего процесса (resident set size) в байтах, или None."""
    try:
        if _SYSTEM == "Windows":
            rss, _ = _win_process_counters()
            return rss
        if _SYSTEM == "Linux":
            rss, _ = _linux_process_status()
            return rss
        if _SYSTEM == "Darwin":
            rss, _ = _macos_process_rss()
            return rss
    except Exception:
        return None
    return None


def get_process_peak_rss():
    """Пиковый RSS текущего процесса в байтах, или None."""
    try:
        if _SYSTEM == "Windows":
            _, peak = _win_process_counters()
            return peak
        if _SYSTEM == "Linux":
            _, peak = _linux_process_status()
            return peak
        if _SYSTEM == "Darwin":
            _, peak = _macos_process_rss()
            return peak
    except Exception:
        return None
    return None


def human_bytes(n):
    """Читаемое представление байт ('1.50 GB') или '—' для None."""
    if n is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


__all__ = [
    "get_system_memory",
    "get_process_rss",
    "get_process_peak_rss",
    "human_bytes",
]