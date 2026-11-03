"""Reactor — pipeline восстановления: freeze → diagnose → fix → unfreeze."""


def freeze_dependents(supervisor, name):
    """Замораживает все модули, которые зависят от name (транзитивно)."""
    deps = supervisor.dependents_of(name)
    frozen = []
    for d in deps:
        if supervisor.freeze(d):
            frozen.append(d)
    return frozen


def run_diagnose(supervisor, name):
    """Вызывает diagnose-функцию модуля."""
    record = supervisor.get_record(name)
    if record is None or record.diagnose is None:
        return None
    try:
        return record.diagnose()
    except Exception as e:
        return {"error": str(e)}


def run_fix(supervisor, name):
    """Вызывает fix-функцию модуля. True — если починил."""
    record = supervisor.get_record(name)
    if record is None or record.fix is None:
        return False
    try:
        result = record.fix()
        return bool(result)
    except Exception:
        return False


def react(supervisor, name):
    """Полный цикл восстановления для упавшего модуля."""
    record = supervisor.get_record(name)
    if record is None:
        from .support_tools.core_spr.exceptions import ModuleNotRegisteredError
        raise ModuleNotRegisteredError(f"Модуль '{name}' не зарегистрирован")

    with supervisor._lock:
        record.consecutive_failures += 1
        record.total_failures += 1

    report = {
        "module": name,
        "frozen": [],
        "diagnosis": None,
        "fix_attempted": False,
        "fix_success": False,
        "unfrozen": [],
        "shutdown_triggered": False,
        "error": record.last_error,
    }

    if record.consecutive_failures > supervisor.max_retries:
        if supervisor.shutdown is not None:
            try:
                supervisor.shutdown.trigger(
                    f"collapse7: {name} exceeded max_retries ({supervisor.max_retries})"
                )
                report["shutdown_triggered"] = True
            except Exception as e:
                report["error"] = f"shutdown failed: {e}"
        return report

    report["frozen"] = freeze_dependents(supervisor, name)
    report["diagnosis"] = run_diagnose(supervisor, name)
    report["fix_attempted"] = (record.fix is not None)
    report["fix_success"] = run_fix(supervisor, name)

    if report["fix_success"]:
        for d in report["frozen"]:
            if supervisor.unfreeze(d):
                report["unfrozen"].append(d)
        supervisor.reset_failures(name)

    return report


__all__ = ["react", "freeze_dependents", "run_diagnose", "run_fix"]