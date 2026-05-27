"""
Weighted SFT Training with Codebook Embedding Injection (Stage 2)
=================================================================
Combines:
  - Per-sample loss weighting (from weighted_sft/train.py)
  - Pre-trained codebook embedding injection (from train_codebook_sft_stage2.py)

Usage:
  export ALIGNED_EMBED_PATH=./data/toys_codebook_aligned_embeddings.safetensors
  export NUM_NEW_TOKENS=1792
  torchrun --nproc_per_node=N train_codebook_weighted_sft.py [llamafactory args...]
"""

import os
import sys
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# 1. Patch AlpacaDatasetConverter to forward the ``weight`` field
# ---------------------------------------------------------------------------
from llamafactory.data.converter import AlpacaDatasetConverter

_orig_alpaca_call = AlpacaDatasetConverter.__call__


def _patched_alpaca_call(self, example: dict[str, Any]) -> dict[str, Any]:
    result = _orig_alpaca_call(self, example)
    result["_loss_weight"] = example.get("weight", 1.0)
    return result


AlpacaDatasetConverter.__call__ = _patched_alpaca_call

# ---------------------------------------------------------------------------
# 2. Patch SupervisedDatasetProcessor to carry ``loss_weight`` into model_inputs
# ---------------------------------------------------------------------------
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX

_orig_preprocess = SupervisedDatasetProcessor.preprocess_dataset


def _patched_preprocess(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
    model_inputs = _orig_preprocess(self, examples)

    if "_loss_weight" in examples:
        kept_weights: list[float] = []
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                continue
            kept_weights.append(float(examples["_loss_weight"][i]))

        assert len(kept_weights) == len(model_inputs["input_ids"]), (
            f"Weight count mismatch: {len(kept_weights)} weights vs {len(model_inputs['input_ids'])} samples"
        )
        model_inputs["loss_weight"] = kept_weights

    return model_inputs


SupervisedDatasetProcessor.preprocess_dataset = _patched_preprocess

# ---------------------------------------------------------------------------
# 3. Weighted data collator
# ---------------------------------------------------------------------------


@dataclass
class WeightedSFTDataCollator:
    """Wraps any base collator, extracting ``loss_weight`` before padding and
    re-injecting it as a float tensor in the output batch."""

    base_collator: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        loss_weights = [f.pop("loss_weight", 1.0) for f in features]
        batch = self.base_collator(features)
        batch["loss_weight"] = torch.tensor(loss_weights, dtype=torch.float32)
        return batch


# ---------------------------------------------------------------------------
# 4. Weighted trainer with per-sample loss weighting
# ---------------------------------------------------------------------------
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


class WeightedSeq2SeqTrainer(CustomSeq2SeqTrainer):
    """CustomSeq2SeqTrainer with per-sample loss weighting."""

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if "loss_weight" not in self._signature_columns:
            self._signature_columns.append("loss_weight")

    def _save(self, output_dir=None, state_dict=None):
        processing_class = self.processing_class
        if isinstance(processing_class, dict):
            self.processing_class = processing_class["tokenizer"]
        super()._save(output_dir, state_dict=state_dict)
        self.processing_class = processing_class

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_weight = inputs.pop("loss_weight", None)
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(shift_logits.size(0), shift_logits.size(1))

        valid_mask = (shift_labels != IGNORE_INDEX).float()

        if loss_weight is not None:
            loss_weight = loss_weight.to(per_token_loss.device).unsqueeze(1)
            weighted_token_loss = per_token_loss * loss_weight
        else:
            weighted_token_loss = per_token_loss

        loss = weighted_token_loss.sum() / (valid_mask.sum() + 1e-8)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# 5. Embedding injection callback
# ---------------------------------------------------------------------------
from transformers import TrainerCallback


class InjectAlignedEmbeddingsCallback(TrainerCallback):
    """Inject pre-trained token embeddings from alignment checkpoint before training."""

    def __init__(self, aligned_embed_path: str, num_new_tokens: int):
        self.aligned_embed_path = aligned_embed_path
        self.num_new_tokens = num_new_tokens

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        from safetensors.torch import load_file

        tensors = load_file(self.aligned_embed_path)
        aligned_embed = tensors['codebook_embeddings']

        if aligned_embed.shape[0] != self.num_new_tokens:
            raise RuntimeError(
                f"Expected {self.num_new_tokens} embeddings, got {aligned_embed.shape[0]} "
                f"in {self.aligned_embed_path}"
            )

        with torch.no_grad():
            embed_weight = model.get_input_embeddings().weight
            embed_weight[-self.num_new_tokens:] = aligned_embed.to(embed_weight.device, embed_weight.dtype)

        print(f"[Stage2] Injected {self.num_new_tokens} aligned embeddings from {self.aligned_embed_path}")


# ---------------------------------------------------------------------------
# 6. Main: weighted SFT workflow with embedding injection
# ---------------------------------------------------------------------------
def run_weighted_codebook_sft():
    from llamafactory.hparams import get_train_args
    from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
    from llamafactory.extras.ploting import plot_loss
    from llamafactory.model import load_model, load_tokenizer
    from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
    from llamafactory.train.trainer_utils import create_ref_model

    aligned_embed_path = os.environ.get("ALIGNED_EMBED_PATH", "./data/toys_codebook_aligned_embeddings.safetensors")
    num_new_tokens = int(os.environ.get("NUM_NEW_TOKENS", "1792"))
    print(f"[Stage2] Will inject {num_new_tokens} aligned embeddings from {aligned_embed_path}")

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    ref_model = None
    if finetuning_args.use_asft_loss:
        ref_model = create_ref_model(model_args, finetuning_args)

    # Build collator
    base_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    data_collator = WeightedSFTDataCollator(base_collator=base_collator)

    # Metrics
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Generation kwargs
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
    if not isinstance(extra_ids, list):
        extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
        string_tokens = [str(t) for t in extra_special_tokens]
        extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
    all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
    gen_kwargs["eos_token_id"] = list(dict.fromkeys(all_eos_ids))
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Embedding injection callback
    inject_callback = InjectAlignedEmbeddingsCallback(
        aligned_embed_path=aligned_embed_path,
        num_new_tokens=num_new_tokens,
    )

    # Trainer
    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        model_args=model_args,
        tokenizer=tokenizer_module,
        data_collator=data_collator,
        callbacks=[inject_callback],
        gen_kwargs=gen_kwargs,
        processor=tokenizer_module.get("processor"),
        ref_model=ref_model,
        **dataset_module,
        **metric_module,
    )

    # Train
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Eval
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)


if __name__ == "__main__":
    run_weighted_codebook_sft()
