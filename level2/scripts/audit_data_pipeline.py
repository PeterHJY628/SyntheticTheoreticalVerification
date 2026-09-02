#!/usr/bin/env python3
"""Sanity checks for Level 2 data generation and AL selection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synthetic.config import ActiveLearningConfig, SpiralConfig
from synthetic.population import generate_active_learning_data


def main() -> None:
    spiral_cfg = SpiralConfig(seed=42)
    al_cfg = ActiveLearningConfig(seed=42, n_pool=200_000)
    data = generate_active_learning_data(spiral_cfg, al_cfg)

    y_pool = data["y_pool_hidden"]
    n0 = int((y_pool == 0).sum())
    n1 = int((y_pool == 1).sum())
    print(f"Pool size: {len(y_pool)} | class 0: {n0} | class 1: {n1}")

    head = y_pool[:10_000]
    print(f"First 10k slice (old viz bug): class0={(head==0).sum()}, class1={(head==1).sum()}")

    # Contiguity check: first 10k should NOT be single-class after shuffle
    if int((head == 1).sum()) == 0:
        print("FAIL: pool still class-segregated at head")
        sys.exit(1)

    y_ec = data["y_eval_center"]
    y_eo = data["y_eval_outer"]
    print(f"Eval center: class0={(y_ec==0).sum()}, class1={(y_ec==1).sum()}")
    print(f"Eval outer:  class0={(y_eo==0).sum()}, class1={(y_eo==1).sum()}")

    # select_topk global check with synthetic scores
    from synthetic.critic import select_topk_by_critic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores_manual = torch.zeros(len(y_pool), device=device)
    scores_manual[y_pool == 1] = 10.0
    scores_manual[y_pool == 0] = 1.0
    kk = 100
    _, idx = torch.topk(scores_manual, k=kk)
    picked = y_pool[idx.cpu()]
    print(f"Manual global top-{kk}: class0={(picked==0).sum()}, class1={(picked==1).sum()}")
    if int((picked == 1).sum()) < kk:
        print("FAIL: global topk logic")
        sys.exit(1)

    # Critic path: scores on shuffled pool, global topk via select_topk_by_critic
    from synthetic.critic import SpectralNormLayerIPM

    x_u = data["x_pool"].to(device)
    p_u = torch.full((len(x_u), 2), 0.5, device=device)
    critic = SpectralNormLayerIPM(4, 8).to(device)
    torch.manual_seed(0)
    for p in critic.parameters():
        p.data.fill_(0.01)
    idx_c = select_topk_by_critic(critic, x_u, p_u, k=50, batch_size=65536)
    picked_c = y_pool[idx_c.cpu()]
    print(f"Critic top-50: class0={(picked_c==0).sum()}, class1={(picked_c==1).sum()}")

    print("OK: data pipeline checks passed")


if __name__ == "__main__":
    main()
