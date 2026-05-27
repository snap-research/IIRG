"""
Training script for alignment stage: freeze backbone, train only new special token embeddings.

This wraps LlamaFactory's standard training with a gradient mask callback
that zeros out gradients for all embedding rows except the newly added tokens.
"""
import sys
import json

from llamafactory.hparams import get_train_args
from llamafactory.train.tuner import _training_function
from llamafactory.train.freeze_embed_callback import FreezeOldEmbeddingsCallback


def main():
    # Detect num_new_tokens from special_tokens file
    # Parse --dataset_dir and dataset name from sys.argv to find token count
    # Or just pass it explicitly via env var or arg
    import os
    num_new_tokens = int(os.environ.get("NUM_NEW_TOKENS", "1536"))

    print(f"[Alignment] Freezing all embeddings except last {num_new_tokens} tokens")

    # Build the callback
    freeze_callback = FreezeOldEmbeddingsCallback(num_new_tokens=num_new_tokens)

    # Run with the callback injected
    config = {
        "args": sys.argv[1:],
        "callbacks": [freeze_callback],
    }
    _training_function(config)

if __name__ == "__main__":
    main()
