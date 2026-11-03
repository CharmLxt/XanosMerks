"""Fusion-правила: склейка соседних операций в одну fused-операцию."""
from .graph import Graph, Node
from .ops import (
    FusedRMSNormLinearOp, FusedLayerNormLinearOp,
    FusedResidualNormOp, FusedSwiGLUOp, FusedLinearSiluOp,
)
from .support_tools.sup2_spr.rules import FUSABLE_PAIRS, ESTIMATED_SPEEDUP


def try_fuse(node_a, node_b):
    """Пробует склеить два узла. Возвращает новый Node или None."""
    pair = (node_a.op_type, node_b.op_type)
    if pair not in FUSABLE_PAIRS:
        return None

    fused_type = FUSABLE_PAIRS[pair]

    if fused_type == "fused_rmsnorm_linear":
        op = FusedRMSNormLinearOp(
            node_a.op.weight, node_b.op.weight, node_b.op.bias, node_a.op.eps,
        )
        inputs = list(node_a.inputs)

    elif fused_type == "fused_layernorm_linear":
        op = FusedLayerNormLinearOp(
            node_a.op.weight, node_a.op.bias,
            node_b.op.weight, node_b.op.bias, node_a.op.eps,
        )
        inputs = list(node_a.inputs)

    elif fused_type == "fused_residual_norm":
        op = FusedResidualNormOp(node_b.op.weight, node_b.op.eps)
        inputs = list(node_a.inputs)

    elif fused_type == "fused_swiglu":
        op = FusedSwiGLUOp()
        if node_b.inputs[0] == node_a.name:
            gate_inp = node_a.inputs[0]
            up_inp = node_b.inputs[1]
        elif node_b.inputs[1] == node_a.name:
            gate_inp = node_a.inputs[0]
            up_inp = node_b.inputs[0]
        else:
            return None
        inputs = [gate_inp, up_inp]

    elif fused_type == "fused_linear_silu":
        op = FusedLinearSiluOp(node_a.op.weight, node_a.op.bias)
        inputs = list(node_a.inputs)

    else:
        return None

    return Node(op, inputs, name=f"fused_{node_a.name}_{node_b.name}")


def apply_fusions(graph):
    """Применяет все применимые fusion-правила к графу."""
    order = graph.topological_order()

    consumers = {name: [] for name in graph.nodes}
    for name, node in graph.nodes.items():
        for inp in node.inputs:
            if inp in graph.nodes:
                consumers[inp].append(name)

    name_map = {inp: inp for inp in graph.inputs}
    fused_nodes = {}
    new_nodes = {}
    report = {"applied": [], "skipped": [], "speedup_estimate": 0.0}

    for name in order:
        if name in fused_nodes:
            continue

        node = graph.nodes[name]
        fused = False

        if len(consumers[name]) == 1:
            consumer_name = consumers[name][0]
            if consumer_name not in fused_nodes:
                consumer = graph.nodes[consumer_name]
                fused_node = try_fuse(node, consumer)

                if fused_node is not None:
                    new_inputs = []
                    for inp in fused_node.inputs:
                        new_inputs.append(name_map.get(inp, inp))

                    new_node = Node(fused_node.op, new_inputs, fused_node.name)
                    new_nodes[fused_node.name] = new_node

                    fused_nodes[name] = fused_node.name
                    fused_nodes[consumer_name] = fused_node.name
                    name_map[name] = fused_node.name
                    name_map[consumer_name] = fused_node.name

                    report["applied"].append(
                        (node.op_type, consumer.op_type, fused_node.op_type)
                    )
                    report["speedup_estimate"] += ESTIMATED_SPEEDUP.get(
                        fused_node.op_type, 0.0
                    )
                    fused = True

        if not fused:
            new_inputs = [name_map.get(inp, inp) for inp in node.inputs]
            new_node = Node(node.op, new_inputs, name)
            new_nodes[name] = new_node
            name_map[name] = name

    new_graph = Graph()
    new_graph.inputs = set(graph.inputs)
    new_graph.nodes = new_nodes
    return new_graph, report


__all__ = ["try_fuse", "apply_fusions"]