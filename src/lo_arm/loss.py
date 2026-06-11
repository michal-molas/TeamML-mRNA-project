import torch
import torch.nn.functional as F

try:
    from .data import build_model_inputs
except ImportError:  # pragma: no cover - script execution fallback
    from data import build_model_inputs


OBJECTIVE_ALPHA_BETA = "alpha_beta"
OBJECTIVE_ELBO = "elbo"
OBJECTIVE_LEGACY_FULL_CANVAS = "legacy_full_canvas"
OBJECTIVE_MODES = {
    OBJECTIVE_ALPHA_BETA,
    OBJECTIVE_ELBO,
    OBJECTIVE_LEGACY_FULL_CANVAS,
}


def ab_schedule(
    step,
    total_steps,
    alpha0=0.025,
    beta0=1.05,
    hold_frac=0.25,
    decay_frac=0.25,
):
    if total_steps <= 0:
        return 0.0, 1.0
    t_hold = hold_frac * total_steps
    t_decay = (hold_frac + decay_frac) * total_steps
    if step < t_hold:
        return float(alpha0), float(beta0)
    if step < t_decay:
        f = (step - t_hold) / max(t_decay - t_hold, 1e-12)
        alpha = alpha0 * (1.0 - f)
        beta = beta0 - (beta0 - 1.0) * f
        return float(alpha), float(beta)
    return 0.0, 1.0


def mask_value_logits(value_logits, mask_id, invalid_value_ids=None):
    logits = value_logits.clone()
    ids = {int(mask_id)}
    if invalid_value_ids is not None:
        ids.update(int(idx) for idx in invalid_value_ids)
    for idx in ids:
        logits[..., idx] = float("-inf")
    return logits


def gumbel_topk_permutation(logits, valid_mask=None):
    if valid_mask is not None:
        logits = logits.masked_fill(~valid_mask, float("-inf"))
    noise = -torch.empty_like(logits).exponential_().log()
    return torch.argsort(logits + noise, dim=-1, descending=True)


def _as_n_previous_tensor(n_previous, order_mask):
    real_lengths = order_mask.sum(dim=-1)
    max_previous = (real_lengths - 1).clamp_min(0)
    if n_previous is None:
        random_unit = torch.rand(order_mask.size(0), device=order_mask.device)
        return torch.floor(random_unit * (max_previous + 1).float()).long()
    if torch.is_tensor(n_previous):
        values = n_previous.to(device=order_mask.device, dtype=torch.long)
        if values.ndim == 0:
            values = values.expand(order_mask.size(0))
    else:
        values = torch.full(
            (order_mask.size(0),), int(n_previous), device=order_mask.device, dtype=torch.long
        )
    return torch.minimum(values.clamp_min(0), max_previous)


def _n_previous_log_value(n_previous):
    if torch.is_tensor(n_previous):
        if n_previous.numel() == 1:
            return int(n_previous.item())
        return float(n_previous.float().mean().item())
    return int(n_previous)


def make_partial_target(target_ids, permutations, n_previous, mask_id, order_mask=None):
    if order_mask is None:
        order_mask = torch.ones_like(target_ids, dtype=torch.bool)
    partial = torch.where(
        order_mask,
        torch.full_like(target_ids, mask_id),
        target_ids,
    )
    n_previous = _as_n_previous_tensor(n_previous, order_mask)
    max_previous = int(n_previous.max().item()) if n_previous.numel() else 0
    for step in range(max_previous):
        active = n_previous > step
        if not active.any():
            continue
        slots = permutations[active, step]
        values = target_ids[active].gather(1, slots.unsqueeze(1)).squeeze(1)
        partial[active, slots] = values
    return partial


def _remaining_mask(permutations, n_previous, target_len, initial_mask=None):
    if initial_mask is None:
        remaining = torch.ones(
            permutations.size(0), target_len, dtype=torch.bool, device=permutations.device
        )
    else:
        remaining = initial_mask.clone()
    n_previous = _as_n_previous_tensor(n_previous, remaining)
    max_previous = int(n_previous.max().item()) if n_previous.numel() else 0
    for step in range(max_previous):
        active = n_previous > step
        if not active.any():
            continue
        remaining[active, permutations[active, step]] = False
    return remaining


def _masked_log_softmax(logits, valid_mask):
    masked = logits.masked_fill(~valid_mask, float("-inf"))
    return F.log_softmax(masked, dim=-1)


def _entropy_from_log_probs(log_probs, valid_mask):
    safe_log_probs = log_probs.masked_fill(~valid_mask, 0.0)
    probs = log_probs.exp().masked_fill(~valid_mask, 0.0)
    return -(probs * safe_log_probs).sum(dim=-1)


def _log_prob_prefix(static_logits, permutations, n_previous, initial_mask):
    n_previous = _as_n_previous_tensor(n_previous, initial_mask)
    if int(n_previous.max().item()) <= 0:
        return torch.zeros(static_logits.size(0), device=static_logits.device)

    remaining = initial_mask.clone()
    log_prob = torch.zeros(static_logits.size(0), device=static_logits.device)
    for step in range(int(n_previous.max().item())):
        active = n_previous > step
        if not active.any():
            continue
        log_probs = _masked_log_softmax(static_logits, remaining)
        selected = permutations[:, step]
        selected_log_prob = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
        log_prob = log_prob + torch.where(active, selected_log_prob, torch.zeros_like(log_prob))
        remaining[active, selected[active]] = False
    return log_prob


def _batch_regions(batch):
    if "prefix_region_ids" not in batch or "target_region_ids" not in batch:
        return None, None
    return batch["prefix_region_ids"], batch["target_region_ids"]


def _model_inputs(batch, target_canvas, target_order_mask=None):
    prefix_region_ids, target_region_ids = _batch_regions(batch)
    built = build_model_inputs(
        batch["prefix_ids"],
        batch["prefix_padding_mask"],
        target_canvas,
        prefix_region_ids=prefix_region_ids,
        target_region_ids=target_region_ids,
        target_order_mask=target_order_mask,
    )
    if len(built) == 2:
        input_ids, padding_mask = built
        region_ids = None
    else:
        input_ids, padding_mask, region_ids = built
    return input_ids, padding_mask, region_ids


def _forward_model(model, input_ids, padding_mask, region_ids):
    if region_ids is None:
        return model(input_ids, padding_mask)
    return model(input_ids, padding_mask, region_ids=region_ids)


def _valid_order_mask(batch, objective_mode):
    target_ids = batch["target_ids"]
    if objective_mode == OBJECTIVE_LEGACY_FULL_CANVAS or "target_order_mask" not in batch:
        return torch.ones_like(target_ids, dtype=torch.bool)
    return batch["target_order_mask"].bool()


def _exact_f_term(
    model,
    batch,
    permutations,
    n_previous,
    q_logits,
    mask_id,
    order_mask,
    alpha,
    beta,
    invalid_value_ids=None,
):
    target_ids = batch["target_ids"]
    partial_target = make_partial_target(
        target_ids,
        permutations,
        n_previous,
        mask_id,
        order_mask=order_mask,
    )
    input_ids, padding_mask, region_ids = _model_inputs(batch, partial_target, order_mask)
    outputs = _forward_model(model, input_ids, padding_mask, region_ids)
    value_logits = mask_value_logits(outputs["value_logits"], mask_id, invalid_value_ids)
    order_logits = outputs["order_logits"]

    remaining = _remaining_mask(permutations, n_previous, target_ids.size(1), order_mask)
    log_q = _masked_log_softmax(q_logits, remaining)
    q_probs = log_q.exp().masked_fill(~remaining, 0.0)
    log_p_order = _masked_log_softmax(order_logits, remaining)

    log_p_values_all = F.log_softmax(value_logits, dim=-1)
    log_p_value = log_p_values_all.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    recon = (q_probs * log_p_value.masked_fill(~remaining, 0.0)).sum(dim=-1)
    distill = (q_probs * log_p_order.masked_fill(~remaining, 0.0)).sum(dim=-1)
    entropy = -(q_probs * log_q.masked_fill(~remaining, 0.0)).sum(dim=-1)
    true_f = recon + distill + entropy
    train_f = recon + beta * distill + (1.0 + alpha) * entropy
    kl = -distill - entropy

    stats = {
        "recon": recon,
        "distill": distill,
        "entropy": entropy,
        "kl": kl,
        "train_f": train_f,
        "true_f": true_f,
        "order_entropy": _entropy_from_log_probs(log_p_order, remaining),
        "posterior_entropy": entropy,
        "value_nll": -recon,
    }
    return train_f, true_f, stats


def compute_lo_arm_loss(
    model,
    batch,
    mask_id,
    n_previous=None,
    objective_mode=OBJECTIVE_ALPHA_BETA,
    alpha=0.0,
    beta=1.0,
    invalid_value_ids=None,
):
    """Return a minimization loss plus detached logging metrics."""

    if objective_mode not in OBJECTIVE_MODES:
        raise ValueError(f"Unsupported objective_mode={objective_mode!r}")
    if objective_mode in {OBJECTIVE_ELBO, OBJECTIVE_LEGACY_FULL_CANVAS}:
        alpha = 0.0
        beta = 1.0
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if beta < 1:
        raise ValueError("beta must be at least 1")

    requested_n_previous = n_previous
    target_ids = batch["target_ids"]
    order_mask = _valid_order_mask(batch, objective_mode)
    if (order_mask.sum(dim=-1) <= 0).any():
        raise ValueError("Every batch item must have at least one orderable target slot")

    full_input_ids, full_padding_mask, full_region_ids = _model_inputs(
        batch, target_ids, order_mask
    )
    full_outputs = _forward_model(model, full_input_ids, full_padding_mask, full_region_ids)
    q_logits = full_outputs["posterior_logits"].masked_fill(~order_mask, float("-inf"))

    n_previous_tensor = _as_n_previous_tensor(n_previous, order_mask)
    perm_1 = gumbel_topk_permutation(q_logits, order_mask).detach()
    perm_2 = gumbel_topk_permutation(q_logits, order_mask).detach()

    f_1, true_f_1, stats_1 = _exact_f_term(
        model,
        batch,
        perm_1,
        n_previous_tensor,
        q_logits,
        mask_id,
        order_mask,
        alpha,
        beta,
        invalid_value_ids=invalid_value_ids,
    )
    f_2, true_f_2, stats_2 = _exact_f_term(
        model,
        batch,
        perm_2,
        n_previous_tensor,
        q_logits,
        mask_id,
        order_mask,
        alpha,
        beta,
        invalid_value_ids=invalid_value_ids,
    )

    log_q_1 = _log_prob_prefix(q_logits, perm_1, n_previous_tensor, order_mask)
    log_q_2 = _log_prob_prefix(q_logits, perm_2, n_previous_tensor, order_mask)
    delta_f = (f_1 - f_2).detach()
    l_real = order_mask.sum(dim=-1).float()
    objective = 0.5 * l_real * ((log_q_1 - log_q_2) * delta_f + f_1 + f_2)
    loss = -objective.mean()

    with torch.no_grad():
        true_objective = 0.5 * l_real * (true_f_1 + true_f_2)
        tilted_objective = 0.5 * l_real * (f_1 + f_2)
        negative_elbo = -true_objective.mean()
        negative_tilted_objective = -tilted_objective.mean()
        order_entropy = 0.5 * (
            stats_1["order_entropy"].mean() + stats_2["order_entropy"].mean()
        )
        posterior_entropy = 0.5 * (
            stats_1["posterior_entropy"].mean() + stats_2["posterior_entropy"].mean()
        )
        value_nll = 0.5 * (stats_1["value_nll"].mean() + stats_2["value_nll"].mean())
        kl_qp = 0.5 * (stats_1["kl"].mean() + stats_2["kl"].mean())
        recon = 0.5 * (stats_1["recon"].mean() + stats_2["recon"].mean())
        distill = 0.5 * (stats_1["distill"].mean() + stats_2["distill"].mean())
        target_length = l_real.mean()

    return {
        "loss": loss,
        "negative_elbo": negative_elbo,
        "negative_tilted_objective": negative_tilted_objective,
        "order_entropy": order_entropy,
        "posterior_entropy": posterior_entropy,
        "value_nll": value_nll,
        "kl_qp": kl_qp,
        "recon": recon,
        "distill": distill,
        "target_length": target_length,
        "alpha": float(alpha),
        "beta": float(beta),
        "n_previous": _n_previous_log_value(
            n_previous_tensor if requested_n_previous is None else requested_n_previous
        ),
        "objective_mode": objective_mode,
    }
