"""gen_configs.py: EXP012 用 108 条件 YAML ファイルを自動生成する。

命名規則:
    {group}_p{perp_scale}_h{patch_h}_w{patch_w}.yaml

    グループ:
        A: patch_t_min=-1.0,  patch_t_max=2.0   （ベースライン）
        B: patch_t_min=-1.0,  patch_t_max=2.25
        C: patch_t_min=-1.25, patch_t_max=2.0
        D: patch_t_min=-1.25, patch_t_max=2.25

使い方:
    python experiments/EXP012_patch_grid_sweep/scr/gen_configs.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "config"

# ベースライン（EXP011 M0 相当）
BASE_CONFIG: dict = {
    "backbone": "custom",
    "loss_fn": "t",
    "use_pq_ratio": False,
    "use_sigmoid": True,
    "use_batchnorm": True,
    "num_conv_layers": 3,
    "use_t_weighted_sampler": False,
    "lr": 1.0e-4,
    "weight_decay": 1.0e-4,
    "epochs": 100,
    "batch": 32,
    "k": 1,
    "patience": 25,
    "seed": 42,
    "aug_flags": 79,
    "label_t_min": -0.5,
    "label_t_max": 1.5,
}

# sweep パラメータ
GROUPS: dict[str, dict[str, float]] = {
    "A": {"patch_t_min": -1.0,  "patch_t_max": 2.0},
    "B": {"patch_t_min": -1.0,  "patch_t_max": 2.25},
    "C": {"patch_t_min": -1.25, "patch_t_max": 2.0},
    "D": {"patch_t_min": -1.25, "patch_t_max": 2.25},
}

PERP_SCALES: list[float] = [0.5, 1.0, 2.0]
PATCH_HEIGHTS: list[int]  = [32, 64, 128]
PATCH_WIDTHS: list[int]   = [64, 128, 256]


def perp_label(v: float) -> str:
    """0.5 → '05', 1.0 → '10', 2.0 → '20'"""
    return f"{int(v * 10):02d}"


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    for group, tm_params in GROUPS.items():
        for ps in PERP_SCALES:
            for ph in PATCH_HEIGHTS:
                for pw in PATCH_WIDTHS:
                    name = f"{group}_p{perp_label(ps)}_h{ph:03d}_w{pw:03d}"
                    comment = (
                        f"# {name}: "
                        f"t_min={tm_params['patch_t_min']}, "
                        f"t_max={tm_params['patch_t_max']}, "
                        f"perp_scale={ps}, "
                        f"h={ph}, w={pw}"
                    )
                    cfg = {
                        **BASE_CONFIG,
                        **tm_params,
                        "patch_perp_scale": ps,
                        "patch_h": ph,
                        "patch_w": pw,
                    }
                    out_path = CONFIG_DIR / f"{name}.yaml"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(comment + "\n")
                        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                                  sort_keys=False)
                    generated.append(name)

    print(f"生成完了: {len(generated)} 条件")
    for n in generated:
        print(f"  {n}")


if __name__ == "__main__":
    main()
