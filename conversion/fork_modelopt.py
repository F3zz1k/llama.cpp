# Fork-only ModelOpt checks, kept out of base.py so upstream rewrites of the dequant code do not
# conflict with them. Called from ModelBase.dequant_model (fork: Inkling).

from __future__ import annotations

from typing import Iterable


def reject_stacked_nvfp4_experts(model_tensors: Iterable[str]) -> None:
    # Stacked-expert NVFP4 checkpoints (e.g. Inkling-NVFP4) store experts as
    # w13_weight/w2_weight with .scale/.scale2/.input_amax/.original_shape
    # auxiliaries; the main tensors do not end in .weight, so the ModelBase NVFP4 path
    # would silently skip them. Reject them with a clear error.
    stacked_expert_aux = [
        n for n in model_tensors
        if n.endswith((".scale2", ".input_amax", ".original_shape"))
    ]
    if stacked_expert_aux:
        raise NotImplementedError(
            "This checkpoint stores quantized experts in the stacked ModelOpt NVFP4 layout "
            f"({len(stacked_expert_aux)} auxiliary tensors like {stacked_expert_aux[0]!r}), "
            "which is not supported yet. Convert from the unquantized (BF16) checkpoint instead."
        )
