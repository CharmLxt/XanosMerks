"""Graph — минимальный вычислительный граф для анализа и fusion."""
from .support_tools.core_spr.exceptions import GraphError


class Node:
    """Узел графа: одна операция + её входы."""

    def __init__(self, op, inputs, name=None):
        self.op = op
        self.inputs = list(inputs)
        self.name = name or f"node_{id(self)}"
        self.output = None

    @property
    def op_type(self):
        return self.op.op_type

    def __repr__(self):
        return f"<Node {self.name} op={self.op_type} inputs={self.inputs}>"


class Graph:
    """Граф вычислений."""

    def __init__(self):
        self.nodes = {}
        self.inputs = set()

    def add_input(self, name):
        self.inputs.add(name)
        return name

    def add_node(self, op, inputs, name=None):
        node = Node(op, inputs, name)
        self.nodes[node.name] = node
        return node.name

    def get_node(self, name):
        if name not in self.nodes:
            raise GraphError(f"Узел '{name}' не найден")
        return self.nodes[name]

    def topological_order(self):
        visited = set()
        order = []

        def dfs(name):
            if name in visited or name in self.inputs:
                return
            visited.add(name)
            node = self.nodes[name]
            for inp in node.inputs:
                dfs(inp)
            order.append(name)

        for name in self.nodes:
            dfs(name)
        return order

    def forward(self, **inputs):
        for inp in self.inputs:
            if inp not in inputs:
                raise GraphError(f"Не передан обязательный вход: {inp}")
        results = dict(inputs)
        for name in self.topological_order():
            node = self.nodes[name]
            args = [results[inp] for inp in node.inputs]
            results[name] = node.op.forward(*args)
            node.output = results[name]
        return results

    def analyze(self):
        from .analyzer import analyze
        return analyze(self)

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"<Graph nodes={len(self.nodes)} inputs={len(self.inputs)}>"


__all__ = ["Node", "Graph"]