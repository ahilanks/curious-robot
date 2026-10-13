"""Rename the released LeWM checkpoint's ViT keys (transformers 4.x names) to the installed transformers 5.x
names, so the authors' eval.py can load it unmodified. Pure key renaming (same tensors, strict load checked);
the remap table is rp1/rp1/wm.py's, which reproduced the paper's CEM numbers on 09-17.

    STABLEWM_HOME=lewm/swm_home .venv/bin/python tools/remap_released_ckpt.py \
        --src swm_home/hf_pusht --dst swm_home/checkpoints/released/lewm_pusht
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from hydra.utils import instantiate

OLD2NEW = [
    (r'encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query', r'encoder.layers.\1.attention.q_proj'),
    (r'encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key', r'encoder.layers.\1.attention.k_proj'),
    (r'encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value', r'encoder.layers.\1.attention.v_proj'),
    (r'encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense', r'encoder.layers.\1.attention.o_proj'),
    (r'encoder\.encoder\.layer\.(\d+)\.intermediate\.dense', r'encoder.layers.\1.mlp.fc1'),
    (r'encoder\.encoder\.layer\.(\d+)\.output\.dense', r'encoder.layers.\1.mlp.fc2'),
    (r'encoder\.encoder\.layer\.(\d+)\.', r'encoder.layers.\1.'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    cfg = json.loads((src / 'config.json').read_text())
    sd = torch.load(src / 'weights.pt', map_location='cpu', weights_only=False)
    out, renamed = {}, 0
    for k, v in sd.items():
        nk = k
        for pat, rep in OLD2NEW:
            nk = re.sub(pat, rep, nk)
        renamed += nk != k
        out[nk] = v
    model = instantiate(cfg)
    model.load_state_dict(out, strict=True)  # raises on any missing / unexpected key
    n_param = sum(p.numel() for p in model.parameters())
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst / 'weights.pt')
    shutil.copy(src / 'config.json', dst / 'config.json')
    print(f'renamed {renamed}/{len(sd)} keys; strict load OK; {n_param / 1e6:.2f}M params -> {dst}')


if __name__ == '__main__':
    main()
