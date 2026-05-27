import torch
import torch.nn.functional as F

try:
    from .data import build_model_inputs
except ImportError:  # pragma: no cover - supports `python train.py`
    from data import build_model_inputs


def mask_value_logits(value_logits, mask_id):
    logits = value_logits.clone()
    logits[..., mask_id] = float("-inf")
    return logits


def gumbel_topk_permutation(logits):
    noise = -torch.empty_like(logits).exponential_().log()
    return torch.argsort(logits + noise, dim=-1, descending=True)


def make_partial_target(target_ids, permutations, n_previous, mask_id):
    partial = torch.full_like(target_ids, mask_id)
    if n_previous <= 0:
        return partial
    previous_slots = permutations[:, :n_previous]
    previous_values = target_ids.gather(1, previous_slots)
    partial.scatter_(1, previous_slots, previous_values)
    return partial


def _remaining_mask(permutations, n_previous, target_len):
    remaining = torch.ones(
        permutations.size(0), target_len, dtype=torch.bool, device=permutations.device
    )
    if n_previous > 0:
        remaining.scatter_(1, permutations[:, :n_previous], False)
    return remaining


def _masked_log_softmax(logits, valid_mask):
    masked = logits.masked_fill(~valid_mask, float("-inf"))
    return F.log_softmax(masked, dim=-1)


def _log_prob_prefix(static_logits, permutations, n_previous):
    if n_previous <= 0:
        return torch.zeros(static_logits.size(0), device=static_logits.device)

    batch_size, target_len = static_logits.shape
    remaining = torch.ones(
        batch_size, target_len, dtype=torch.bool, device=static_logits.device
    )
    log_prob = torch.zeros(batch_size, device=static_logits.device)
    for step in range(n_previous):
        log_probs = _masked_log_softmax(static_logits, remaining)
        selected = permutations[:, step]
        log_prob = log_prob + log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
        remaining.scatter_(1, selected.unsqueeze(1), False)
    return log_prob


def _exact_f_term(model, batch, permutations, n_previous, q_logits, mask_id):
    target_ids = batch["target_ids"]
    partial_target = make_partial_target(target_ids, permutations, n_previous, mask_id)
    input_ids, padding_mask = build_model_inputs(
        batch["prefix_ids"], batch["prefix_padding_mask"], partial_target
    )
    outputs = model(input_ids, padding_mask)
    value_logits = mask_value_logits(outputs["value_logits"], mask_id)
    order_logits = outputs["order_logits"]

    remaining = _remaining_mask(permutations, n_previous, target_ids.size(1))
    log_q = _masked_log_softmax(q_logits, remaining)
    q_probs = log_q.exp()
    log_p_order = _masked_log_softmax(order_logits, remaining)

    log_p_values_all = F.log_softmax(value_logits, dim=-1)
    log_p_value = log_p_values_all.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    term = log_p_order + log_p_value - log_q
    return (q_probs * term.masked_fill(~remaining, 0.0)).sum(dim=-1)


def compute_lo_arm_loss(model, batch, mask_id, n_previous=None):
    """Return a minimization loss plus detached logging metrics.

    This implements the two-sample RLOO autodiff objective from LO-ARM. The
    returned loss is the negative stochastic ELBO objective.
    """

    target_ids = batch["target_ids"]
    full_input_ids, full_padding_mask = build_model_inputs(
        batch["prefix_ids"], batch["prefix_padding_mask"], target_ids
    )
    full_outputs = model(full_input_ids, full_padding_mask)
    q_logits = full_outputs["posterior_logits"]

    target_len = target_ids.size(1)
    if n_previous is None:
        n_previous = int(torch.randint(0, target_len, (1,), device=target_ids.device).item())

    perm_1 = gumbel_topk_permutation(q_logits).detach()
    perm_2 = gumbel_topk_permutation(q_logits).detach()

    f_1 = _exact_f_term(model, batch, perm_1, n_previous, q_logits, mask_id)
    f_2 = _exact_f_term(model, batch, perm_2, n_previous, q_logits, mask_id)

    log_q_1 = _log_prob_prefix(q_logits, perm_1, n_previous)
    log_q_2 = _log_prob_prefix(q_logits, perm_2, n_previous)
    delta_f = (f_1 - f_2).detach()
    objective = 0.5 * target_len * ((log_q_1 - log_q_2) * delta_f + f_1 + f_2)
    loss = -objective.mean()

    with torch.no_grad():
        negative_elbo = -(0.5 * target_len * (f_1 + f_2)).mean()

    return {
        "loss": loss,
        "negative_elbo": negative_elbo,
        "n_previous": n_previous,
    }
