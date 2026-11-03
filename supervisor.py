"""Collapse7 — супервизор 7 модулей XanosMerks."""
import functools
import threading
import time


_SENTINEL = object()


class ModuleStatus:
    """Статусы модуля."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    FROZEN = "frozen"
    DEGRADED = "degraded"


class ModuleRecord:
    """Запись о зарегистрированном модуле."""

    def __init__(self, name, health_check=None, diagnose=None, fix=None,
                 depends_on=None):
        self.name = name
        self.health_check = health_check
        self.diagnose = diagnose
        self.fix = fix
        self.depends_on = tuple(depends_on or ())
        self.status = ModuleStatus.OK
        self.consecutive_failures = 0
        self.total_failures = 0
        self.last_error = None
        self.last_check_time = None
        self._lock = threading.RLock()

    def reset_failures(self):
        with self._lock:
            self.consecutive_failures = 0
            self.status = ModuleStatus.OK

    def __repr__(self):
        return (
            f"<ModuleRecord {self.name} status={self.status} "
            f"fails={self.consecutive_failures}/{self.total_failures}>"
        )


class _FrozenGuard:
    """Контекстный менеджер: падает, если модуль заморожен."""

    def __init__(self, supervisor, name):
        self.supervisor = supervisor
        self.name = name

    def __enter__(self):
        if self.supervisor.is_frozen(self.name):
            self.supervisor._handle_frozen_call(self.name)
        return self

    def __exit__(self, *exc):
        return False


class Collapse7:
    """Супервизор модулей."""

    def __init__(self, max_retries=3, shutdown=None, on_frozen_call=None):
        """
        max_retries: сколько раз подряд можно пробовать fix до shutdown
        shutdown: msisp.GracefulShutdown или None
        on_frozen_call: callback(name) | None
            Вызывается перед raise Collapse7FrozenError, если модуль заморожен.
            Удобно для логирования.
        """
        self.max_retries = max_retries
        self.shutdown = shutdown
        self.on_frozen_call = on_frozen_call
        self._records = {}
        self._lock = threading.RLock()

    def _handle_frozen_call(self, name):
        """Действие при попытке вызвать замороженный модуль.

        Если on_frozen_call — callable, вызывается (для логирования),
        затем всегда raise Collapse7FrozenError.
        """
        from .support_tools.core_spr.exceptions import Collapse7FrozenError

        handler = self.on_frozen_call
        if callable(handler):
            try:
                handler(name)
            except Exception:
                pass
        raise Collapse7FrozenError(
            f"Модуль '{name}' заморожен супервизором collapse7"
        )

    # ---------- Регистрация ----------

    def register(self, name, health_check=None, diagnose=None, fix=None,
                 depends_on=None):
        with self._lock:
            record = ModuleRecord(
                name=name,
                health_check=health_check,
                diagnose=diagnose,
                fix=fix,
                depends_on=depends_on,
            )
            self._records[name] = record
        return name

    def unregister(self, name):
        with self._lock:
            if name in self._records:
                del self._records[name]
                return True
            return False

    def list_modules(self):
        with self._lock:
            return list(self._records.keys())

    # ---------- Проверки ----------

    def check(self, name):
        with self._lock:
            record = self._records.get(name)
            if record is None:
                from .support_tools.core_spr.exceptions import ModuleNotRegisteredError
                raise ModuleNotRegisteredError(f"Модуль '{name}' не зарегистрирован")

        status = ModuleStatus.OK
        error = None

        if record.health_check is not None:
            try:
                result = record.health_check()
                if result is True or result == ModuleStatus.OK or result is None:
                    status = ModuleStatus.OK
                elif result == ModuleStatus.WARNING:
                    status = ModuleStatus.WARNING
                elif result == ModuleStatus.ERROR:
                    status = ModuleStatus.ERROR
                else:
                    status = ModuleStatus.WARNING
            except Exception as e:
                status = ModuleStatus.ERROR
                error = str(e)

        with self._lock:
            record.status = status
            record.last_error = error
            record.last_check_time = time.time()
            if status == ModuleStatus.ERROR:
                record.consecutive_failures += 1
                record.total_failures += 1

        return status

    def check_all(self):
        with self._lock:
            names = list(self._records.keys())
        return {name: self.check(name) for name in names}

    # ---------- Freeze ----------

    def freeze(self, name):
        with self._lock:
            record = self._records.get(name)
            if record is None:
                return False
            if record.status != ModuleStatus.FROZEN:
                record.status = ModuleStatus.FROZEN
            return True

    def unfreeze(self, name):
        with self._lock:
            record = self._records.get(name)
            if record is None:
                return False
            if record.status == ModuleStatus.FROZEN:
                record.status = ModuleStatus.OK
            return True

    def unfreeze_all(self):
        """Снимает freeze со всех замороженных модулей.

        Возвращает список имён, с которых freeze был снят.
        """
        with self._lock:
            names = [
                n for n, r in self._records.items()
                if r.status == ModuleStatus.FROZEN
            ]
        for n in names:
            self.unfreeze(n)
        return names

    def is_frozen(self, name):
        with self._lock:
            record = self._records.get(name)
            return record is not None and record.status == ModuleStatus.FROZEN

    def frozen_modules(self):
        """Список имён замороженных модулей."""
        with self._lock:
            return [
                n for n, r in self._records.items()
                if r.status == ModuleStatus.FROZEN
            ]

    def dependents_of(self, name):
        """Возвращает список модулей, которые зависят от name (транзитивно)."""
        with self._lock:
            graph = {n: r.depends_on for n, r in self._records.items()}

        dependents = []
        visited = set()

        def dfs(target, path):
            if target in path:
                from .support_tools.core_spr.exceptions import CircularDependencyError
                raise CircularDependencyError(f"Цикл: {' -> '.join(path + [target])}")
            for mod, deps in graph.items():
                if target in deps and mod not in visited:
                    visited.add(mod)
                    dependents.append(mod)
                    dfs(mod, path + [target])

        dfs(name, [])
        return dependents

    # ---------- Защита вызова ----------

    def protected(self, name, default=_SENTINEL):
        """Декоратор: если модуль заморожен — raise Collapse7FrozenError.

        Параметры:
            name: имя модуля в супервизоре.
            default: если задан — при заморозке вернуть это значение,
                а не бросать исключение.

        Пример:
            @supervisor.protected("nsofi40")
            def generate(...):
                ...

            @supervisor.protected("meanflare", default=None)
            def try_load(...):
                ...
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self.is_frozen(name):
                    if default is not _SENTINEL:
                        return default
                    self._handle_frozen_call(name)
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def guard(self, name):
        """Контекстный менеджер: raise Collapse7FrozenError, если заморожен.

        Пример:
            with supervisor.guard("nsofi40"):
                generator.generate(...)
        """
        return _FrozenGuard(self, name)

    # ---------- Реакция на ошибку ----------

    def handle_failure(self, name):
        from .reactor import react
        return react(self, name)

    def reset_failures(self, name):
        with self._lock:
            record = self._records.get(name)
            if record is None:
                return False
            record.reset_failures()
            return True

    # ---------- Состояние ----------

    def get_record(self, name):
        with self._lock:
            return self._records.get(name)

    def status(self):
        with self._lock:
            records = dict(self._records)
        modules = {}
        frozen = []
        errors = 0
        warnings = 0
        for name, r in records.items():
            modules[name] = {
                "status": r.status,
                "consecutive_failures": r.consecutive_failures,
                "total_failures": r.total_failures,
                "last_error": r.last_error,
            }
            if r.status == ModuleStatus.FROZEN:
                frozen.append(name)
            elif r.status == ModuleStatus.ERROR:
                errors += 1
            elif r.status == ModuleStatus.WARNING:
                warnings += 1
        return {
            "modules": modules,
            "frozen": frozen,
            "errors": errors,
            "warnings": warnings,
        }

    def __repr__(self):
        with self._lock:
            return f"<Collapse7 modules={len(self._records)} max_retries={self.max_retries}>"


__all__ = ["Collapse7", "ModuleRecord", "ModuleStatus"]