"""
Callback to freeze all embedding rows except the last `num_new_tokens`.
This ensures only the newly added special token embeddings receive gradient updates.

Usage:
    In your training script, register this callback:
        callbacks = [FreezeOldEmbeddingsCallback(num_new_tokens=1536)]
"""
from transformers import TrainerCallback


class FreezeOldEmbeddingsCallback(TrainerCallback):
    """Zero out gradients for all embedding rows except the last `num_new_tokens`."""

    def __init__(self, num_new_tokens: int):
        self.num_new_tokens = num_new_tokens
        self._hooks = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        def make_mask_hook(n_new):
            def hook(grad):
                grad[:-n_new] = 0.0
                return grad
            return hook

        hook_fn = make_mask_hook(self.num_new_tokens)

        # Input embeddings
        inp_emb = model.get_input_embeddings().weight
        self._hooks.append(inp_emb.register_hook(hook_fn))

        # Output head (lm_head) — skip if tied with input embeddings
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.data_ptr() != inp_emb.data_ptr():
            self._hooks.append(out_emb.weight.register_hook(hook_fn))

    def on_train_end(self, args, state, control, **kwargs):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
