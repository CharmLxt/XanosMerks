"""MoE (Mixture of Experts) слой для Mixtral и подобных моделей."""
import numpy as np

from .primitives import fast_softmax, apply_activation


class MoELayer:
    """Sparse Mixture of Experts.

    Ожидает nested dict:
        weights["gate"]["weight"]                     # (num_experts, hidden)
        weights["experts"][str(i)]["w1"]["weight"]   # (intermediate, hidden)
        weights["experts"][str(i)]["w2"]["weight"]   # (hidden, intermediate)
        weights["experts"][str(i)]["w3"]["weight"]   # (intermediate, hidden)
    """

    def __init__(self, config, weights):
        self.num_experts = config["num_experts"]
        self.num_experts_per_tok = config["num_experts_per_tok"]
        self.activation = config.get("hidden_act", "silu")

        if self.num_experts_per_tok < 1:
            raise ValueError(
                f"MoELayer: num_experts_per_tok={self.num_experts_per_tok} < 1"
            )
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"MoELayer: num_experts_per_tok={self.num_experts_per_tok} "
                f"> num_experts={self.num_experts}"
            )

        gate_w = weights["gate"]["weight"].astype(np.float32)
        if gate_w.ndim != 2:
            raise ValueError(
                f"MoELayer: gate.weight должен быть 2D, получено {gate_w.ndim}D"
            )
        self.gate_T = gate_w.T

        self.experts_w1_T = []
        self.experts_w2_T = []
        self.experts_w3_T = []

        for i in range(self.num_experts):
            exp = weights["experts"][str(i)]
            w1 = exp["w1"]["weight"].astype(np.float32)
            w2 = exp["w2"]["weight"].astype(np.float32)
            w3 = exp["w3"]["weight"].astype(np.float32)
            self.experts_w1_T.append(w1.T)
            self.experts_w2_T.append(w2.T)
            self.experts_w3_T.append(w3.T)

        self._is_dense = (self.num_experts_per_tok == self.num_experts)

    def __call__(self, x):
        """x: (batch, seq, hidden) -> (batch, seq, hidden)."""
        batch, seq, hidden = x.shape
        x_flat = x.reshape(-1, hidden)
        n_tokens = x_flat.shape[0]
        k = self.num_experts_per_tok

        router_logits = x_flat @ self.gate_T

        if self._is_dense:
            weights = fast_softmax(router_logits, axis=-1)
            output = np.zeros_like(x_flat)
            for exp_idx in range(self.num_experts):
                gate = x_flat @ self.experts_w1_T[exp_idx]
                up = x_flat @ self.experts_w3_T[exp_idx]
                hidden_act = apply_activation(gate, self.activation) * up
                out = hidden_act @ self.experts_w2_T[exp_idx]
                output += out * weights[:, exp_idx:exp_idx + 1]
            return output.reshape(batch, seq, hidden)

        # Sparse: top-k, sort-based dispatch.
        top_k_indices = np.argpartition(
            -router_logits, k - 1, axis=-1
        )[:, :k]
        top_k_logits = np.take_along_axis(
            router_logits, top_k_indices, axis=-1
        )
        top_k_weights = fast_softmax(top_k_logits, axis=-1)

        flat_experts = top_k_indices.reshape(-1)
        flat_weights = top_k_weights.reshape(-1)
        flat_token_idx = np.repeat(np.arange(n_tokens), k)

        sort_order = np.argsort(flat_experts, kind="stable")
        sorted_experts = flat_experts[sort_order]
        sorted_weights = flat_weights[sort_order]
        sorted_token_idx = flat_token_idx[sort_order]

        experts_counts = np.bincount(sorted_experts, minlength=self.num_experts)
        cumsum = np.cumsum(experts_counts)

        output = np.zeros_like(x_flat)
        start = 0
        for exp_idx in range(self.num_experts):
            end = int(cumsum[exp_idx])
            count = end - start
            if count == 0:
                continue

            token_slice = sorted_token_idx[start:end]
            weight_slice = sorted_weights[start:end]

            tokens_sel = x_flat[token_slice]
            gate = tokens_sel @ self.experts_w1_T[exp_idx]
            up = tokens_sel @ self.experts_w3_T[exp_idx]
            hidden_act = apply_activation(gate, self.activation) * up
            out = hidden_act @ self.experts_w2_T[exp_idx]

            output[token_slice] += out * weight_slice[:, None]
            start = end

        return output.reshape(batch, seq, hidden)

    def __repr__(self):
        mode = "dense" if self._is_dense else f"top-{self.num_experts_per_tok}"
        return f"<MoELayer experts={self.num_experts} {mode} activation={self.activation}>"


__all__ = ["MoELayer"]