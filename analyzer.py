"""Analyzer — отчёт по графу/слою/модели: паттерны, оценки, память."""
from .support_tools.core_spr.exceptions import UnsupportedOpError
from .support_tools.sup2_spr.rules import FUSABLE_PAIRS, ESTIMATED_SPEEDUP


def _analyze_graph(graph):
    consumers = {name: [] for name in graph.nodes}
    for name, node in graph.nodes.items():
        for inp in node.inputs:
            if inp in graph.nodes:
                consumers[inp].append(name)

    pattern_counts = {}
    for name, node in graph.nodes.items():
        if len(consumers[name]) == 1:
            consumer = graph.nodes[consumers[name][0]]
            pair = (node.op_type, consumer.op_type)
            if pair in FUSABLE_PAIRS:
                pattern_counts[pair] = pattern_counts.get(pair, 0) + 1

    patterns = []
    total_speedup = 0.0
    for pair, count in pattern_counts.items():
        fused_as = FUSABLE_PAIRS[pair]
        speedup = ESTIMATED_SPEEDUP.get(fused_as, 0.0)
        patterns.append({
            "ops": list(pair),
            "count": count,
            "fusable_as": fused_as,
            "estimated_speedup": speedup,
        })
        total_speedup += count * speedup

    return {
        "target_type": "graph",
        "total_ops": len(graph.nodes),
        "patterns": patterns,
        "estimated_total_speedup": total_speedup,
        "heavy_ops": 0,
        "notes": [],
    }


def _analyze_optimized_layer(layer):
    patterns = []
    speedup = 0.0

    attn = layer.attn
    if getattr(attn, "use_fused", False):
        fused_as = (
            "fused_rmsnorm_linear"
            if attn.norm_type == "rmsnorm"
            else "fused_layernorm_linear"
        )
        patterns.append({
            "ops": [attn.norm_type, "linear"],
            "count": 1,
            "fusable_as": fused_as,
            "estimated_speedup": ESTIMATED_SPEEDUP.get(fused_as, 0.05),
        })
        speedup += ESTIMATED_SPEEDUP.get(fused_as, 0.05)

    if getattr(layer, "use_fused_mlp", False):
        patterns.append({
            "ops": ["rmsnorm", "linear"],
            "count": 1,
            "fusable_as": "fused_rmsnorm_linear",
            "estimated_speedup": ESTIMATED_SPEEDUP.get("fused_rmsnorm_linear", 0.05),
        })
        speedup += ESTIMATED_SPEEDUP.get("fused_rmsnorm_linear", 0.05)

        patterns.append({
            "ops": ["silu", "mul"],
            "count": 1,
            "fusable_as": "fused_swiglu",
            "estimated_speedup": ESTIMATED_SPEEDUP.get("fused_swiglu", 0.02),
        })
        speedup += ESTIMATED_SPEEDUP.get("fused_swiglu", 0.02)

    if getattr(layer, "use_fused_residual", False):
        patterns.append({
            "ops": ["add", "rmsnorm"],
            "count": 1,
            "fusable_as": "fused_residual_norm",
            "estimated_speedup": ESTIMATED_SPEEDUP.get("fused_residual_norm", 0.03),
        })
        speedup += ESTIMATED_SPEEDUP.get("fused_residual_norm", 0.03)

    return {
        "target_type": "optimized_layer",
        "total_ops": 5,
        "patterns": patterns,
        "estimated_total_speedup": speedup,
        "heavy_ops": 0,
        "notes": [],
    }


def _analyze_optimized_model(model):
    per_layer = [_analyze_optimized_layer(l) for l in model.layers]
    total_ops = sum(r["total_ops"] for r in per_layer)
    total_speedup = sum(r["estimated_total_speedup"] for r in per_layer)

    all_patterns = []
    for r in per_layer:
        all_patterns.extend(r["patterns"])

    return {
        "target_type": "optimized_model",
        "total_ops": total_ops,
        "patterns": all_patterns,
        "estimated_total_speedup": total_speedup,
        "heavy_ops": 0,
        "notes": [f"{len(model.layers)} layers"],
    }


def analyze(target):
    """Анализирует граф, слой или модель."""
    from .graph import Graph
    from .optimizer import OptimizedLayer, OptimizedModel

    if isinstance(target, Graph):
        return _analyze_graph(target)
    if isinstance(target, OptimizedLayer):
        return _analyze_optimized_layer(target)
    if isinstance(target, OptimizedModel):
        return _analyze_optimized_model(target)

    raise UnsupportedOpError(f"analyze: unsupported target type {type(target)}")


__all__ = ["analyze"]