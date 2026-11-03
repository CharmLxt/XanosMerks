"""GracefulShutdown — упорядоченное выполнение callback'ов при завершении."""
import threading
import warnings


def _run_with_timeout(callback, timeout):
    """Запускает callback в отдельном потоке. Возвращает (success, error, timed_out)."""
    done = threading.Event()
    result = {"error": None}

    def _run():
        try:
            callback()
        except Exception as e:
            result["error"] = e
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    finished = done.wait(timeout)
    if not finished:
        return False, None, True
    return result["error"] is None, result["error"], False


class GracefulShutdown:
    """Упорядоченное graceful завершение."""

    def __init__(self, timeout_per_callback=5.0):
        self.timeout_per_callback = timeout_per_callback
        self._handlers = []
        self._order = 0
        self._lock = threading.RLock()
        self._triggered = False
        self._in_progress = False
        self._orig_signals = {}

    def register(self, callback, priority=5, name=None):
        with self._lock:
            if name is None:
                name = f"handler_{self._order}"
            self._handlers.append((priority, self._order, callback, name))
            self._order += 1
            self._handlers.sort(key=lambda x: (x[0], x[1]))
            return name

    def unregister(self, name):
        with self._lock:
            for i, (p, o, cb, n) in enumerate(self._handlers):
                if n == name:
                    del self._handlers[i]
                    return True
            return False

    def has_handler(self, name):
        with self._lock:
            return any(n == name for _p, _o, _cb, n in self._handlers)

    def clear(self):
        with self._lock:
            self._handlers.clear()

    def trigger(self, reason="manual"):
        with self._lock:
            if self._in_progress:
                return False
            if self._triggered:
                return False
            self._triggered = True
            self._in_progress = True
            handlers = list(self._handlers)

        all_ok = True
        for p, o, cb, name in handlers:
            ok, err, timed_out = _run_with_timeout(cb, self.timeout_per_callback)
            if timed_out:
                warnings.warn(
                    f"GracefulShutdown: handler '{name}' не завершился за "
                    f"{self.timeout_per_callback} сек"
                )
                all_ok = False
            elif not ok:
                warnings.warn(f"GracefulShutdown: handler '{name}' упал: {err}")
                all_ok = False

        with self._lock:
            self._in_progress = False
        return all_ok

    def reset(self):
        with self._lock:
            self._triggered = False
            self._in_progress = False

    @property
    def is_triggered(self):
        with self._lock:
            return self._triggered

    @property
    def is_in_progress(self):
        with self._lock:
            return self._in_progress

    def install_signal_handlers(self, signals=None):
        import signal
        if signals is None:
            signals = [signal.SIGINT]

        def _handler(signum, frame):
            self.trigger(f"signal:{signum}")
            for sig in signals:
                try:
                    orig = self._orig_signals.get(sig, signal.SIG_DFL)
                    signal.signal(sig, orig)
                except Exception:
                    pass

        for sig in signals:
            try:
                self._orig_signals[sig] = signal.getsignal(sig)
                signal.signal(sig, _handler)
            except Exception:
                pass

    def uninstall_signal_handlers(self):
        import signal
        for sig, orig in list(self._orig_signals.items()):
            try:
                signal.signal(sig, orig)
            except Exception:
                pass
        self._orig_signals.clear()

    def __repr__(self):
        return (
            f"<GracefulShutdown handlers={len(self._handlers)} "
            f"triggered={self.is_triggered} in_progress={self.is_in_progress}>"
        )


__all__ = ["GracefulShutdown"]