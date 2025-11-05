from transformers import TrainerCallback
from trl import SFTTrainer
import torch
import torch.nn as nn


class KLLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print("#"*20 + " KLLoggerCallback ")
        print(logs)
        print("#"*20)
        if logs and "kl_loss" in logs:
            print(f"[step {state.global_step}] CE = {logs['ce_loss']:.4f} - KL = {logs['kl_loss']:.4f} - Total = {logs['loss']:.4f}")

class SFTTrainerWithKL(SFTTrainer):
    def __init__(self, *args, kl_lambda=0.01, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_lambda = kl_lambda
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    @torch.no_grad()
    def _logits_ref_without_lora(self, inputs_no_labels):
        try:
            with self.model.disable_adapter():
                outputs_ref = self.model(**inputs_no_labels)
        except Exception:
            was_training = self.model.training
            if hasattr(self.model, "disable_adapter"):
                self.model.disable_adapter()
            outputs_ref = self.model(**inputs_no_labels)
            if hasattr(self.model, "enable_adapter"):
                self.model.enable_adapter()
            if was_training:
                self.model.train()
        return outputs_ref.logits

    def _kl_pt_pref(self, logits_t, logits_ref, labels):
        # Shift para alinear con CE
        shift_t = logits_t[:, :-1, :] / self.temperature
        shift_ref = logits_ref[:, :-1, :] / self.temperature
        shift_labels = labels[:, 1:]

        valid = (shift_labels != -100).float()
        denom = valid.sum().clamp_min(1.0)

        log_pt = torch.log_softmax(shift_t, dim=-1)
        log_pref = torch.log_softmax(shift_ref, dim=-1)
        pt = log_pt.exp()

        kl_tokens = (pt * (log_pt - log_pref)).sum(dim=-1)  # [B, T-1]
        kl_mean = (kl_tokens * valid).sum() / denom
        return kl_mean

    # Acepta el arg extra de Unsloth
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # 1) Separa labels y NO se los pases al modelo
        labels = inputs.get("labels")
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}

        # 2) Forward normal (LoRA activo) -> logits_t
        outputs_t = model(**inputs_no_labels)
        logits_t = outputs_t.logits

        # 3) CE manual (evita la fused CE de Unsloth)
        shift_logits = logits_t[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_ce = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # 4) Forward de referencia (LoRA desactivado, sin gradiente)
        with torch.no_grad():
            logits_ref = self._logits_ref_without_lora(inputs_no_labels)

        # 5) KL promedio en posiciones válidas
        kl = self._kl_pt_pref(logits_t, logits_ref, labels)

        total = loss_ce + self.kl_lambda * kl

        if self.state is not None and self.state.global_step % self.args.logging_steps == 0:
            self.log({"kl_loss": kl.detach().float().item(), "ce_loss": loss_ce.detach().float().item()})
            
        return (total, outputs_t) if return_outputs else total