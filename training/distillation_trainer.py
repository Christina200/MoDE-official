import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def tokenwise_kl_topk_chunked(student_logits, teacher_logits, *,
                              T=1.0, K=256, chunk_size=64, attn_mask=None):
    """KL( P_t^K || P_s^K ), with both distributions renormalized on the teacher's top-K support.

    Restricting to the teacher's top-K and chunking over the sequence keeps the
    [B, S, V] logits from being materialized as probabilities all at once. Renormalizing
    both sides on the same support makes the result non-negative by construction.
    """
    assert student_logits.shape == teacher_logits.shape
    B, S, V = student_logits.shape

    total = student_logits.new_zeros(())
    denom = 0

    for start in range(0, S, chunk_size):
        end = min(S, start + chunk_size)
        # Do numerics in float32 even if inputs are bf16/fp16
        s_chunk = (student_logits[:, start:end, :].float() / T)
        t_chunk = (teacher_logits[:, start:end, :].float() / T)

        with torch.no_grad():
            K_eff = min(K, V)
            idx = torch.topk(t_chunk, k=K_eff, dim=-1).indices
            t_logp_topk = torch.gather(F.log_softmax(t_chunk, dim=-1), -1, idx)
            t_logpK = t_logp_topk - torch.logsumexp(t_logp_topk, dim=-1, keepdim=True)

        s_logp_topk = torch.gather(F.log_softmax(s_chunk, dim=-1), -1, idx)
        s_logpK = s_logp_topk - torch.logsumexp(s_logp_topk, dim=-1, keepdim=True)

        kl_tokens = (torch.exp(t_logpK) * (t_logpK - s_logpK)).sum(dim=-1)

        if attn_mask is not None:
            m = attn_mask[:, start:end].to(kl_tokens.dtype)
            total = total + (kl_tokens * m).sum()
            denom += m.sum().item()
        else:
            total = total + kl_tokens.sum()
            denom += kl_tokens.numel()

    return (total / max(denom, 1)) * (T * T)


class DistillationTrainer(Trainer):
    """Trainer that adds a self-distillation term on the image-generation stream.

    The teacher is the same model with the MoDE experts disabled, i.e. the pre-trained
    base model, so no separate teacher checkpoint has to be held in memory.
    """

    def __init__(self, lambda_distill=0.5, **kwargs):
        super().__init__(**kwargs)
        self.lambda_distill = lambda_distill

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, temperature=2.0):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
            return (loss, outputs) if return_outputs else loss

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        modality_ids = inputs.get("token_type_ids", None)
        has_image_tokens = modality_ids is not None and torch.sum(modality_ids).item() > 0
        has_img_gen_batch = inputs.get("img_gen_input_ids", None) is not None

        if not (has_image_tokens and has_img_gen_batch):
            loss = ce_loss
            return (loss, outputs) if return_outputs else loss

        if hasattr(model, "module"):
            model.module.set_mode('train-image')
        else:
            model.mode = 'train-image'

        # Teacher pass: experts disabled, so this is the frozen pre-trained model.
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model.base_model._set_teacher_mode(True)
                teacher_outputs = model.base_model(
                    input_ids=inputs["img_gen_input_ids"],
                    attention_mask=inputs.get("img_gen_attention_mask", None),
                )
                teacher_logits = teacher_outputs.logits / temperature
                model.base_model._set_teacher_mode(False)

                del teacher_outputs
                torch.cuda.empty_cache()

        student_img_outputs = model(
            input_ids=inputs["img_gen_input_ids"],
            attention_mask=inputs.get("img_gen_attention_mask", None),
            token_type_ids=inputs.get("img_gen_token_type_ids", None),
            task_id=inputs.get("task_id", -1),
        )
        student_logits = student_img_outputs.logits  # keep original dtype
        del student_img_outputs
        student_logits = student_logits / temperature

        if hasattr(model, "module"):
            model.module.set_mode('train-text')
        else:
            model.mode = 'train-text'

        kd_loss = tokenwise_kl_topk_chunked(
            student_logits, teacher_logits,
            T=temperature,
            K=256,
            chunk_size=16,
        )
        loss = ce_loss + self.lambda_distill * kd_loss

        return (loss, outputs) if return_outputs else loss
