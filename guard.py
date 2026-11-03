"""OOMGuard — proactive (ensure) и reactive (catch_oom) защита от OOM."""
import functools
import threading

from .monitor import MemoryMonitor
from .support_tools.core_spr.exceptions import OOMGuardError


_OOM_EXCEPTIONS = (MemoryError,)
try:
    from numpy._core._exceptions import _ArrayMemoryError as _NumpyArrayMemoryError
    _OOM_EXCEPTIONS = _OOM_EXCEPTIONS + (_NumpyArrayMemoryError,)
except ImportError:
    try:
        from numpy.core._exceptions import _ArrayMemoryError as _NumpyArrayMemoryError
        _OOM_EXCEPTIONS = _OOM_EXCEPTIONS + (_NumpyArrayMemoryError,)
    except ImportError:
        pass


class OOMGuard:
    """Guard, который защищает от OOM."""

    def __init__(self, monitor=None, safety_margin=0.1):
        self.monitor = monitor or MemoryMonitor()
        self.safety_margin = safety_margin
        self._evacuations = []
        self._order = 0
        self._lock = threading.RLock()
        self._shutdown = None

    def set_shutdown(self, shutdown):
        self._shutdown = shutdown

    def register_evacuation(self, callback, priority=5, name=None):
        with self._lock:
            if name is None:
                name = f"evac_{self._order}"
            self._evacuations.append((priority, self._order, callback, name))
            self._order += 1
            self._evacuations.sort(key=lambda x: (x[0], x[1]))
            return name

    def unregister_evacuation(self, name):
        with self._lock:
            for i, (p, o, cb, n) in enumerate(self._evacuations):
                if n == name:
                    del self._evacuations[i]
                    return True
            return False

    def list_evacuations(self):
        with self._lock:
            return [(p, n) for p, o, cb, n in self._evacuations]

    def _try_evacuate(self):
        with self._lock:
            evacs = list(self._evacuations)
        count = 0
        errors = []
        for p, o, cb, name in evacs:
            try:
                cb()
                count += 1
            except Exception as e:
                errors.append((name, e))
        return count, errors

    def ensure(self, nbytes, on_oom="raise"):
        return EnsureContext(self, nbytes, on_oom)

    def catch_oom(self, on_oom="raise", max_attempts=1):
        return CatchOOMContext(self, on_oom, max_attempts)

    def watch(self, func=None, *, on_oom="raise", max_attempts=1):
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                attempt = 0
                while True:
                    try:
                        return fn(*args, **kwargs)
                    except _OOM_EXCEPTIONS as e:
                        attempt += 1
                        if attempt >= max_attempts:
                            self._handle_oom(e, on_oom)
                            raise OOMGuardError(
                                f"OOM после {attempt} попыток: {e}"
                            ) from e
                        self._try_evacuate()
            return wrapper
        if func is not None:
            return decorator(func)
        return decorator

    def _handle_oom(self, exc, on_oom):
        if on_oom == "shutdown" and self._shutdown is not None:
            self._shutdown.trigger(f"OOM: {exc}")
        elif on_oom == "evacuate":
            self._try_evacuate()
        raise OOMGuardError(f"OOM: {exc}") from exc

    def __repr__(self):
        with self._lock:
            n = len(self._evacuations)
        return f"<OOMGuard evacuations={n} margin={self.safety_margin}>"


class EnsureContext:
    def __init__(self, guard, nbytes, on_oom):
        self.guard = guard
        self.nbytes = nbytes
        self.on_oom = on_oom

    def __enter__(self):
        can_fit = self.guard.monitor.check_available(
            self.nbytes, self.guard.safety_margin
        )
        if can_fit is not False:
            return self
        self.guard._try_evacuate()
        can_fit = self.guard.monitor.check_available(
            self.nbytes, self.guard.safety_margin
        )
        if can_fit is not False:
            return self
        if self.on_oom == "shutdown" and self.guard._shutdown is not None:
            self.guard._shutdown.trigger(f"OOM: ensure({self.nbytes})")
        elif self.on_oom == "evacuate":
            return self
        raise OOMGuardError(
            f"ensure({self.nbytes}): недостаточно памяти даже после эвакуации. "
            f"available={self.guard.monitor.available()}"
        )

    def __exit__(self, *exc):
        return False


class CatchOOMContext:
    def __init__(self, guard, on_oom, max_attempts):
        self.guard = guard
        self.on_oom = on_oom
        self.max_attempts = max_attempts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            return False
        if not isinstance(exc, _OOM_EXCEPTIONS):
            return False
        self.guard._try_evacuate()
        if self.on_oom == "evacuate":
            return True
        if self.on_oom == "shutdown" and self.guard._shutdown is not None:
            self.guard._shutdown.trigger(f"OOM: {exc}")
        raise OOMGuardError(f"OOM: {exc}") from exc


__all__ = ["OOMGuard", "EnsureContext", "CatchOOMContext"]