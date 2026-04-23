"""Отрисовка графов TRT-движков через trex с кастомной sparse-раскраской.

Свой `sparse_formatter`: ярко-зелёным подсвечиваются conv-слои, в которых TRT
реально выбрал sparse Tensor Core kernel (имя тактики содержит `sparse_conv` /
`sparse_int8`). Остальные слои раскрашены по precision (FP16/INT8/FP32).
Так в одной картинке видно и precision, и где именно сработало 2:4.
"""
from pathlib import Path

from trex import EnginePlan
from trex.graphing import precision_colormap, render_dot, to_dot

EXP = Path("experiments/trex_sparse")

PLANS = {
    "dense_fp16":  (EXP / "dense_fp16_layers.json",  EXP / "dense_fp16_profile.json"),
    "dense_int8":  (EXP / "dense_int8_layers.json",  EXP / "dense_int8_profile.json"),
    "sparse_fp16": (EXP / "sparse_fp16_layers.json", EXP / "sparse_fp16_profile.json"),
    "sparse_int8": (EXP / "sparse_int8_layers.json", EXP / "sparse_int8_profile.json"),
}


def sparse_formatter(layer):
    tactic = str(layer.raw_dict.get("TacticName", "")).lower()
    uses_sparse_kernel = (
        layer.type == "Convolution"
        and ("sparse_conv" in tactic or "sparse_int8" in tactic)
    )
    fill = "#39FF14" if uses_sparse_kernel else precision_colormap[layer.precision]
    return {
        "shape": "Mrecord",
        "style": "filled",
        "tooltip": layer.tooltip(),
        "fillcolor": fill,
        "color": "lightgray",
        "fontname": "Helvetica",
    }


def main():
    for name, (graph_json, profile_json) in PLANS.items():
        plan = EnginePlan(str(graph_json), str(profile_json), name=f"unet_{name}")
        dot = to_dot(
            plan,
            layer_node_formatter=sparse_formatter,
            display_regions=True,
            display_edge_details=True,
        )
        png = render_dot(dot, str(EXP / f"unet_{name}"), "png")
        print(f"{name}: {png}")


if __name__ == "__main__":
    main()
