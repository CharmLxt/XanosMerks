"""MsispRegistry — глобальный реестр MSISP и стандартные хендлеры."""
import threading

from .monitor import MemoryMonitor
from .guard import OOMGuard
from .shutdown import GracefulShutdown
from .support_tools.sup2_spr.priorities import PRIORITY_HIGH


class MsispRegistry:
    """Глобальный реестр MSISP."""

    def __init__(self):
        self.monitor = None
        self.guard = None
        self.shutdown = None
        self.kv_caches = []
        self.memory_managers = []
        self.generators = []
        self._lock = threading.RLock()

    def register_kv_cache(self, cache):
        with self._lock:
            if cache not in self.kv_caches:
                self.kv_caches.append(cache)

    def register_memory_manager(self, memory_manager):
        with self._lock:
            if memory_manager not in self.memory_managers:
                self.memory_managers.append(memory_manager)

    def register_generator(self, generator, state_path=None):
        with self._lock:
            self.generators.append((generator, state_path))

    def unregister(self, obj):
        with self._lock:
            removed = False
            if obj in self.kv_caches:
                self.kv_caches.remove(obj)
                removed = True
            if obj in self.memory_managers:
                self.memory_managers.remove(obj)
                removed = True
            for i, (g, _sp) in enumerate(self.generators):
                if g is obj:
                    del self.generators[i]
                    removed = True
                    break
            return removed

    def evacuate_all(self):
        with self._lock:
            kv = list(self.kv_caches)
            mm = list(self.memory_managers)
        kv_reset = 0
        mm_unloaded = 0
        errors = []
        for c in kv:
            try:
                c.reset()
                kv_reset += 1
            except Exception as e:
                errors.append(("kv_cache", str(e)))
        for m in mm:
            try:
                m.unload_all()
                mm_unloaded += 1
            except Exception as e:
                errors.append(("memory_manager", str(e)))
        return {"kv_reset": kv_reset, "mm_unloaded": mm_unloaded, "errors": errors}

    def snapshot(self):
        with self._lock:
            n_kv = len(self.kv_caches)
            n_mm = len(self.memory_managers)
            n_gen = len(self.generators)
        result = {}
        if self.monitor is not None:
            result["memory"] = self.monitor.snapshot().to_dict()
        result["registered"] = {
            "kv_caches": n_kv,
            "memory_managers": n_mm,
            "generators": n_gen,
        }
        return result

    def __repr__(self):
        with self._lock:
            return (
                f"<MsispRegistry kv={len(self.kv_caches)} "
                f"mm={len(self.memory_managers)} "
                f"gen={len(self.generators)}>"
            )


_REGISTRY = None
_REGISTRY_LOCK = threading.Lock()


def get_registry():
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = MsispRegistry()
        return _REGISTRY


def install_default_handlers(registry=None, kv_caches=None, memory_managers=None):
    """Настраивает MSISP «из коробки»."""
    if registry is None:
        registry = get_registry()

    if registry.monitor is None:
        registry.monitor = MemoryMonitor()
    if registry.shutdown is None:
        registry.shutdown = GracefulShutdown()
    if registry.guard is None:
        registry.guard = OOMGuard(
            monitor=registry.monitor, safety_margin=0.1,
        )
        registry.guard.set_shutdown(registry.shutdown)

    if kv_caches:
        for c in kv_caches:
            registry.register_kv_cache(c)
    if memory_managers:
        for mm in memory_managers:
            registry.register_memory_manager(mm)

    existing_guard = {n for _p, n in registry.guard.list_evacuations()}
    if "registry_evacuate" not in existing_guard:
        registry.guard.register_evacuation(
            registry.evacuate_all,
            priority=PRIORITY_HIGH,
            name="registry_evacuate",
        )

    if not registry.shutdown.has_handler("evacuate_all"):
        registry.shutdown.register(
            registry.evacuate_all,
            priority=PRIORITY_HIGH,
            name="evacuate_all",
        )

    try:
        registry.shutdown.install_signal_handlers()
    except Exception:
        pass

    return registry


__all__ = ["MsispRegistry", "get_registry", "install_default_handlers"]