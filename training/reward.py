import torch

def compute_reward(a_pre, a_intra, a_post):
    """
    Global reward function.
    Higher = better outcome.
    """

    # Example components (can refine later)

    # Pre: UF target stability (penalize extreme values)
    uf_target = a_pre[:, 0]
    pre_penalty = torch.mean((uf_target - 2.5) ** 2)

    # Intra: stability (penalize large adjustments)
    intra_adjust = a_intra[:, 0]
    intra_penalty = torch.mean(intra_adjust ** 2)

    # Post: recovery (encourage small adjustments)
    post_adjust = a_post[:, 0]
    post_penalty = torch.mean(post_adjust ** 2)

    # Combine
    reward = - (pre_penalty + intra_penalty + post_penalty)

    return reward