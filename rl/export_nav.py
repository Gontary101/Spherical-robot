"""
Export a navigation checkpoint (.pt) to the TorchScript actor the evaluators load.

    python rl/export_nav.py rl/runs/nav2s/m007M.pt rl/runs/nav2s/m007M.ts
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nav2_env import ACT_OBS, Nav2Env  # noqa: E402
from train_nav2 import make_agent  # noqa: E402

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    env = Nav2Env(seed=0, low_ckpt=os.path.join(os.path.dirname(__file__), "runs/loco31/best.pt"))
    _, oc = env.reset()
    agent = make_agent(len(oc))
    agent.load(src, optimizer=False)
    agent.export(dst, ACT_OBS)
    print("exported", dst)
