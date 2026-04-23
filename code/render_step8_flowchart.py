from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SVG_PATH = ROOT / "step8_flowchart_main.svg"
HTML_PATH = ROOT / "step8_flowchart.html"


def box(x, y, w, h, title, lines=None, fill="#f8fafc", stroke="#334155"):
    lines = lines or []
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
        f'<text x="{x + w / 2}" y="{y + 24}" text-anchor="middle" class="title">{escape(title)}</text>',
    ]
    for idx, line in enumerate(lines):
        out.append(
            f'<text x="{x + w / 2}" y="{y + 48 + idx * 18}" text-anchor="middle" class="body">{escape(line)}</text>'
        )
    return "\n".join(out)


def diamond(cx, cy, w, h, text):
    points = [
        (cx, cy - h / 2),
        (cx + w / 2, cy),
        (cx, cy + h / 2),
        (cx - w / 2, cy),
    ]
    point_text = " ".join(f"{x},{y}" for x, y in points)
    return "\n".join(
        [
            f'<polygon points="{point_text}" fill="#fff7ed" stroke="#c2410c" stroke-width="1.4"/>',
            f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" class="title">{escape(text)}</text>',
            f'<text x="{cx}" y="{cy + 16}" text-anchor="middle" class="body">通过?</text>',
        ]
    )


def arrow(x1, y1, x2, y2, label=None):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    out = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="1.35" marker-end="url(#arrow)"/>'
    ]
    if label:
        out.append(
            f'<text x="{mid_x}" y="{mid_y - 6}" text-anchor="middle" class="edge">{escape(label)}</text>'
        )
    return "\n".join(out)


def build_svg():
    left_x = 72
    check_x = 438
    fail_x = 755
    start_y = 70
    step = 122
    box_w = 300
    box_h = 82
    diamond_w = 132
    diamond_h = 72

    modules = [
        ("Input", ["database_goal / discipline", "query_requirements / key_description"]),
        ("Load Context", ["读取查询需求与 key_description", "构造 shared_context"]),
        ("Locating Module", ["确定检索单元", "确定组织主轴"]),
        ("Mechanism Requirement", ["抽取领域必备概念", "识别机制约束"]),
        ("Query Semantics", ["把查询需求映射为", "可筛选对象和比较轴"]),
        ("Evidence Model", ["定义证据层级", "建立 figure ownership 原则"]),
        ("Subjective Supervisor", ["确定建模立场", "给出反模板约束"]),
        ("Topic Adaptation", ["判断主题类型", "生成专题化调整原则"]),
        ("Section Partition", ["划分 core sections", "划分 non-core sections"]),
        ("Field Planning", ["规划字段组", "定义证据策略"]),
        ("Figure Supervisor", ["判断是否启用", "figure classification"]),
        ("Figure Classification", ["建立 section-figure", "类别边界"]),
        ("Schema Design", ["生成 top_level_keys", "生成 field_registry"]),
        ("Specialization Critic", ["检查是否仍像通用模板", "规范化 pass / needs_redesign"]),
        ("Aggregation", ["聚合最终 result", "section_design / schema_definition"]),
        ("Postprocess", ["补 skyrmion 关键字段", "自动补 figure_constraint"]),
        ("Validate", ["结构校验 + 专项校验", "生成 validation_errors"]),
        ("Write Output", ["写出 JSON", "success 或 needs_review"]),
    ]

    height = start_y + (len(modules) - 1) * step + 170
    width = 1040
    items = []

    items.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    items.append(
        """
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#475569"/>
  </marker>
  <style>
    .title { font: 700 15px "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; fill: #0f172a; }
    .body { font: 13px "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; fill: #334155; }
    .edge { font: 12px "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; fill: #64748b; }
    .heading { font: 800 24px "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; fill: #0f172a; }
    .sub { font: 14px "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; fill: #475569; }
  </style>
</defs>
<rect width="100%" height="100%" fill="#f1f5f9"/>
<text x="520" y="34" text-anchor="middle" class="heading">Step 8 Section Design Agent 主流程图</text>
<text x="520" y="56" text-anchor="middle" class="sub">从输入上下文到模块调用、后处理、最终校验与 JSON 输出</text>
"""
    )

    fail_y = start_y + 2 * step
    items.append(box(fail_x, fail_y, 210, 86, "Early Stop", ["模块校验失败", "status = needs_review"], "#fee2e2", "#dc2626"))

    previous_bottom = None
    for idx, (title, lines) in enumerate(modules):
        y = start_y + idx * step
        fill = "#dcfce7" if title == "Write Output" else "#f8fafc"
        stroke = "#16a34a" if title == "Write Output" else "#334155"
        items.append(box(left_x, y, box_w, box_h, title, lines, fill, stroke))

        if previous_bottom is not None:
            items.append(arrow(left_x + box_w / 2, previous_bottom, left_x + box_w / 2, y))

        if 2 <= idx <= 14:
            cy = y + box_h / 2
            items.append(diamond(check_x, cy, diamond_w, diamond_h, "校验"))
            items.append(arrow(left_x + box_w, cy, check_x - diamond_w / 2, cy, "输出"))
            items.append(arrow(check_x + diamond_w / 2, cy, fail_x, fail_y + 43, "否"))
            if idx < len(modules) - 1:
                next_y = start_y + (idx + 1) * step
                items.append(arrow(check_x, cy + diamond_h / 2, left_x + box_w / 2, next_y, "是"))
                previous_bottom = None
                continue

        previous_bottom = y + box_h

    note_x = 715
    note_y = start_y + 9 * step
    items.append(box(note_x, note_y, 270, 178, "关键后处理", [
        "normalize_module_result",
        "infer_figure_constraint",
        "detect_skyrmion_section_id",
        "ensure_field",
        "validate_result",
        "validate_specialization",
    ], "#e0f2fe", "#0284c7"))
    items.append(arrow(note_x, note_y + 178, left_x + box_w, start_y + 15 * step + 41, "补齐约束"))

    items.append("</svg>")
    return "\n".join(items)


def build_html():
    md = (ROOT / "step8_flowchart.md").read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Step 8 Section Design Agent Flowchart</title>
  <style>
    body {{ margin: 0; padding: 32px; background: #f8fafc; color: #0f172a; font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; background: white; padding: 28px; border-radius: 18px; box-shadow: 0 18px 50px rgba(15, 23, 42, .12); }}
    h1 {{ margin-top: 0; }}
    .hint {{ color: #64748b; margin-bottom: 24px; }}
    .mermaid {{ margin: 28px 0; padding: 18px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; overflow: auto; }}
    pre {{ white-space: pre-wrap; }}
  </style>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' }});
  </script>
</head>
<body>
<main>
<h1>Step 8 Section Design Agent 流程图</h1>
<p class="hint">如果 Mermaid 图没有显示，请确认浏览器可以访问 cdn.jsdelivr.net。静态主流程图见 step8_flowchart_main.svg。</p>
{markdown_to_html_blocks(md)}
</main>
</body>
</html>
"""


def markdown_to_html_blocks(md):
    blocks = []
    in_mermaid = False
    buffer = []
    for line in md.splitlines():
        if line.strip() == "```mermaid":
            in_mermaid = True
            buffer = []
            continue
        if in_mermaid and line.strip() == "```":
            blocks.append(f'<div class="mermaid">\n{escape(chr(10).join(buffer))}\n</div>')
            in_mermaid = False
            continue
        if in_mermaid:
            buffer.append(line)
        elif line.startswith("# "):
            blocks.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            blocks.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("- "):
            blocks.append(f"<p>• {escape(line[2:])}</p>")
        elif line.strip():
            blocks.append(f"<p>{escape(line)}</p>")
    return "\n".join(blocks)


def main():
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    HTML_PATH.write_text(build_html(), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
