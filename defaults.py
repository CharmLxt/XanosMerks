"""Встроенные регистрации для 7 модулей XanosMerks."""
import importlib
import sys


XANOSMERKS_MODULES = (
    "gamtype",
    "meanflare",
    "genof11mo",
    "nsofi40",
    "msisp",
    "vofcf32",
    "quacdualc",
)

DEFAULT_DEPENDS_ON = {
    "gamtype": (),
    "meanflare": ("gamtype",),
    "genof11mo": ("gamtype", "meanflare", "nsofi40"),
    "nsofi40": (),
    "msisp": (),
    "vofcf32": ("meanflare",),
    "quacdualc": ("genof11mo",),
}


def _module(name):
    return sys.modules.get(f"xanosmerks.{name}")


# ============================================================
# HEALTH CHECKS
# ============================================================

def _health_gamtype():
    import numpy as np
    from xanosmerks.gamtype.readers import _normalize_model_type
    assert _normalize_model_type("llama") == "llama"
    assert _normalize_model_type(None) == "unknown"
    from xanosmerks.gamtype.analysis import detect_architecture_class
    arch = detect_architecture_class({"model_type": "llama"})
    assert arch == "LlamaForCausalLM"
    return "ok"


def _health_meanflare():
    import numpy as np
    from xanosmerks.meanflare.support_tools.core_spr.dtype_ops import (
        bf16_to_fp32, fp32_to_bf16, normalize_dtype_name,
    )
    arr_bf16 = np.array([0x3F80, 0x4000], dtype=np.uint16)
    arr_fp32 = bf16_to_fp32(arr_bf16)
    assert arr_fp32.shape == (2,)
    assert np.allclose(arr_fp32, [1.0, 2.0], atol=1e-3)
    back = fp32_to_bf16(arr_fp32)
    assert back.dtype == np.uint16
    assert normalize_dtype_name("f16") == "float16"
    return "ok"


def _health_genof11mo():
    import numpy as np
    from xanosmerks.genof11mo.primitives import rms_norm, silu
    x = np.ones((1, 4), dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    out = rms_norm(x, w)
    assert out.shape == (1, 4)
    assert np.allclose(out, 1.0, atol=1e-3)
    s = silu(np.array([0.0], dtype=np.float32))
    assert np.allclose(s, [0.0])
    return "ok"


def _health_nsofi40():
    import numpy as np
    from xanosmerks.nsofi40 import NsofiKVCache
    c = NsofiKVCache(1, 1, 4, 8, window_size=4, sink_size=0)
    k = np.ones((1, 1, 1, 4), dtype=np.float32)
    c.append(0, k, k)
    c.advance(1)
    assert c.cur_len == 1
    got_k, _ = c.get(0)
    assert got_k.shape == (1, 1, 1, 4)
    return "ok"


def _health_msisp():
    from xanosmerks.msisp import MemoryMonitor
    m = MemoryMonitor()
    snap = m.snapshot()
    assert snap is not None
    assert hasattr(snap, "to_dict")
    return "ok"


def _health_vofcf32():
    import numpy as np
    from xanosmerks.vofcf32 import Quantizer
    q = Quantizer()
    x = np.random.randn(4, 8).astype(np.float32)
    q_arr, scales, zp = q.quantize_int8(x, scheme="per_tensor")
    assert q_arr.dtype == np.int8
    assert q_arr.shape == x.shape
    restored = q.dequantize_int8(q_arr, scales, zp)
    assert restored.shape == x.shape
    return "ok"


def _health_quacdualc():
    import numpy as np
    from xanosmerks.quacdualc import RMSNormOp, LinearOp
    x = np.ones((1, 4), dtype=np.float32)
    norm = RMSNormOp(np.ones(4)).forward(x)
    assert norm.shape == (1, 4)
    lin = LinearOp(np.eye(4)).forward(norm)
    assert lin.shape == (1, 4)
    return "ok"


HEALTH_CHECKS = {
    "gamtype": _health_gamtype,
    "meanflare": _health_meanflare,
    "genof11mo": _health_genof11mo,
    "nsofi40": _health_nsofi40,
    "msisp": _health_msisp,
    "vofcf32": _health_vofcf32,
    "quacdualc": _health_quacdualc,
}


# ============================================================
# DIAGNOSE
# ============================================================

def _diagnose_generic(name):
    def _diag():
        mod = _module(name)
        if mod is None:
            return {"module": name, "loaded": False, "error": "not in sys.modules"}
        return {
            "module": name,
            "loaded": True,
            "version": getattr(mod, "__version__", None),
        }
    return _diag


def _diagnose_nsofi40(registry):
    def _diag():
        mod = _module("nsofi40")
        if mod is None:
            return {"module": "nsofi40", "loaded": False}
        result = {"module": "nsofi40", "loaded": True}
        if registry is not None:
            result["registered_caches"] = len(registry.kv_caches)
        return result
    return _diag


def _diagnose_meanflare(registry):
    def _diag():
        mod = _module("meanflare")
        if mod is None:
            return {"module": "meanflare", "loaded": False}
        result = {"module": "meanflare", "loaded": True}
        if registry is not None:
            result["registered_managers"] = len(registry.memory_managers)
            total_used = 0
            total_max = 0
            for mm in registry.memory_managers:
                try:
                    s = mm.summary()
                    total_used += s["used_bytes"]
                    if s["max_bytes"]:
                        total_max += s["max_bytes"]
                except Exception:
                    pass
            result["total_used_bytes"] = total_used
            result["total_max_bytes"] = total_max
        return result
    return _diag


DIAGNOSES = {
    "gamtype": _diagnose_generic("gamtype"),
    "meanflare": None,
    "genof11mo": _diagnose_generic("genof11mo"),
    "nsofi40": None,
    "msisp": _diagnose_generic("msisp"),
    "vofcf32": _diagnose_generic("vofcf32"),
    "quacdualc": _diagnose_generic("quacdualc"),
}


# ============================================================
# FIX
# ============================================================

def _fix_reload(name):
    def _fix():
        mod = _module(name)
        if mod is None:
            return False
        try:
            importlib.reload(mod)
            return True
        except Exception:
            return False
    return _fix


def _fix_nsofi40_reset(registry):
    def _fix():
        if registry is None:
            return False
        try:
            for c in list(registry.kv_caches):
                c.reset()
            return True
        except Exception:
            return False
    return _fix


def _fix_meanflare_unload(registry):
    def _fix():
        if registry is None:
            return False
        try:
            for mm in list(registry.memory_managers):
                mm.unload_all()
            return True
        except Exception:
            return False
    return _fix


def _fix_msisp(registry):
    def _fix():
        mod = _module("msisp")
        if mod is None:
            return False
        try:
            if registry is not None and registry.monitor is not None:
                try:
                    registry.monitor.stop()
                except Exception:
                    pass
            importlib.reload(mod)
            return True
        except Exception:
            return False
    return _fix


FIXES = {
    "gamtype": _fix_reload("gamtype"),
    "meanflare": None,
    "genof11mo": None,
    "nsofi40": None,
    "msisp": None,
    "vofcf32": _fix_reload("vofcf32"),
    "quacdualc": _fix_reload("quacdualc"),
}


# ============================================================
# REGISTRATION
# ============================================================

def register_defaults(supervisor, registry=None):
    """Регистрирует все 7 модулей XanosMerks в переданном супервизоре.

    Параметры:
        supervisor: Collapse7
        registry: msisp.MsispRegistry | None
            Если передан, fix для nsofi40/meanflare/msisp использует его
            (reset всех кэшей / unload_all / перезапуск монитора).
            Если None, эти модули получают fix = None.
    """
    diagnoses = dict(DIAGNOSES)
    diagnoses["nsofi40"] = _diagnose_nsofi40(registry)
    diagnoses["meanflare"] = _diagnose_meanflare(registry)

    fixes = dict(FIXES)
    fixes["nsofi40"] = _fix_nsofi40_reset(registry)
    fixes["meanflare"] = _fix_meanflare_unload(registry)
    fixes["msisp"] = _fix_msisp(registry)

    for name in XANOSMERKS_MODULES:
        supervisor.register(
            name=name,
            health_check=HEALTH_CHECKS.get(name),
            diagnose=diagnoses.get(name),
            fix=fixes.get(name),
            depends_on=DEFAULT_DEPENDS_ON.get(name, ()),
        )
    return supervisor


__all__ = [
    "XANOSMERKS_MODULES",
    "DEFAULT_DEPENDS_ON",
    "HEALTH_CHECKS",
    "register_defaults",
]