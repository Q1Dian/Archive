"""Synthetic episodes for testing the memory core before real features exist.

Each episode has a pool of P random unit contents and its own transition map
pi (a random cyclic order over the pool). The stream is piecewise constant:
segments of 3-8 steps, and the next segment's content is pi(current) with
probability p_follow, otherwise random. So what follows a boundary is
episode-specific: the cue alone cannot predict it, a memory of earlier
transitions in the same episode can.
"""
import torch
import torch.nn.functional as F


def make_episodes(n_episodes, steps, dim=32, pool=6, p_follow=0.9, obs_noise=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    contents = F.normalize(torch.randn(n_episodes, pool, dim, generator=g), dim=-1)
    order = torch.argsort(torch.rand(n_episodes, pool, generator=g), dim=1)
    nxt = torch.empty_like(order)
    nxt.scatter_(1, order, torch.roll(order, -1, dims=1))

    item = torch.empty(n_episodes, steps, dtype=torch.long)
    cur = torch.randint(pool, (n_episodes,), generator=g)
    left = torch.randint(3, 9, (n_episodes,), generator=g)
    for t in range(steps):
        item[:, t] = cur
        left -= 1
        done = left == 0
        follow = torch.rand(n_episodes, generator=g) < p_follow
        rand_item = torch.randint(pool, (n_episodes,), generator=g)
        new = torch.where(follow, nxt.gather(1, cur[:, None])[:, 0], rand_item)
        cur = torch.where(done, new, cur)
        left = torch.where(done, torch.randint(3, 9, (n_episodes,), generator=g), left)

    z = contents.gather(1, item[..., None].expand(-1, -1, dim))
    x = z + obs_noise * torch.randn(z.shape, generator=g) / dim ** 0.5
    boundary = torch.zeros(n_episodes, steps, dtype=torch.bool)
    boundary[:, 1:] = item[:, 1:] != item[:, :-1]
    # a boundary whose (previous, next) transition already occurred earlier in the episode
    repeat = torch.zeros_like(boundary)
    for e in range(n_episodes):
        seen = set()
        for t in torch.nonzero(boundary[e])[:, 0].tolist():
            pair = (item[e, t - 1].item(), item[e, t].item())
            repeat[e, t] = pair in seen
            seen.add(pair)
    return dict(x=x, z=z, item=item, boundary=boundary, repeat=repeat)
