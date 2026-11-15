"""Architecture diagrams for the Phase 3 report.

Each diagram is written as Graphviz ``dot`` source (editable), rendered to SVG, PDF and
300-dpi PNG, and converted to a draw.io file that reuses Graphviz's own layout
coordinates (``dot -Tjson``), so the ``.drawio`` version opens with the same geometry and
can be edited by hand.

Style: white background; rounded boxes; pastel fills (blue = data/physical, green =
twin/virtual, peach = decision, purple = evaluation); bold layer headers; dashed
"detail" boxes; coloured arrows (blue data flow, green predictions, orange decisions,
purple dashed feedback); yellow closed-loop label; sans-serif font.

Usage: python diagrams/make_diagrams.py   (needs Graphviz ``dot`` on PATH or in
C:/Program Files/Graphviz/bin)
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
DOT = shutil.which("dot") or r"C:\Program Files\Graphviz\bin\dot.exe"

# ---------------------------------------------------------------- palette
C = {
    "data": ("#DCEBFA", "#3F75B0"), "data_h": ("#A9CBEF", "#2B5C94"),
    "twin": ("#DDF1DF", "#3F8E4C"), "twin_h": ("#A8DBB0", "#2C6E37"),
    "dec": ("#FDE3C8", "#D27A1F"), "dec_h": ("#F7C08A", "#A85A0E"),
    "eval": ("#ECE1F6", "#7B55A6"), "eval_h": ("#CDB5E8", "#5C3A86"),
    "net": ("#EEF1F4", "#5B6B7A"), "loop": ("#FFF1A8", "#C99A06"),
    "detail": ("#FFFFFF", "#8C8C8C"), "start": ("#CDEFD2", "#3F8E4C"),
}
ARROW = {"data": "#2F6DB5", "pred": "#3C9A4B", "dec": "#E07B00", "fb": "#7E57C2", "plain": "#555555"}
FONT = "Helvetica"
FONT_B = "Helvetica-Bold"

HEADER = f'fontname="{FONT}" fontsize=13 bgcolor="white" pad=0.3 nodesep=0.35 ranksep=0.45 splines=ortho ' \
         f'node [fontname="{FONT}" fontsize=12 shape=box style="rounded,filled" penwidth=1.4 margin="0.18,0.10"] ' \
         f'edge [fontname="{FONT}" fontsize=10.5 penwidth=1.6 arrowsize=0.8]'


def box(nid: str, label: str, kind: str, **extra) -> str:
    fill, edge = C[kind]
    attrs = {"label": label, "fillcolor": fill, "color": edge}
    attrs.update(extra)
    return f'  {nid} [' + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "];"


def header(nid: str, label: str, kind: str, width: float = 0) -> str:
    extra = {"fontname": FONT_B, "fontsize": "14"}
    if width:
        extra.update({"width": str(width), "fixedsize": "false"})
    return box(nid, label, kind, **extra)


def detail(nid: str, title: str, bullets: list[str]) -> str:
    body = f"{title}\\l" + "".join(f"•  {b}\\l" for b in bullets)
    return box(nid, body, "detail", style="rounded,dashed,filled", fontsize="10.5", penwidth="1.2")


def arrow(a: str, b: str, kind: str, label: str = "", **extra) -> str:
    attrs = {"color": ARROW[kind], "fontcolor": ARROW[kind]}
    if label:
        attrs["xlabel"] = f" {label} "
    if kind == "fb":
        attrs["style"] = "dashed"
    attrs.update(extra)
    return f"  {a} -> {b} [" + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "];"


# ---------------------------------------------------------------- diagram 1
def d1_architecture() -> str:
    return "\n".join([
        f"digraph arch {{ {HEADER} rankdir=TB newrank=true compound=true",
        '  label="Three-layer closed-loop architecture: AI-driven network digital twin for safe decision testing" '
        'labelloc=t fontsize=16 fontname="Helvetica-Bold"',
        # ---- data layer
        '  subgraph cluster_data { label="" style="rounded,filled" fillcolor="#F4F9FE" color="#3F75B0" penwidth=1.6',
        header("h_data", "PHYSICAL / DATA LAYER", "data_h"),
        box("teal", "Teal repository\\nB4 real TMs · UsCarrier, Kdl (toy)", "data"),
        box("figret", "FIGRET repository\\nGEANT 22/72 real TMs", "data"),
        box("zoo", "Topology Zoo\\nCogentco + gravity TMs", "data"),
        box("prep", "Preprocessing\\nvalidation · Yen k=4 paths · normalisation\\nlink–path bipartite graph · splits",
            "data", fontname=FONT_B),
        detail("d_data", "Data pipeline details", [
            "topology validation, capacity units checked",
            "Yen k = 4 min-hop candidate paths per OD pair",
            "capacities and demands ÷ max link capacity",
            "link–path bipartite graph (GNN input)",
            "train / val / held-aside time splits; GEANT, Kdl unseen",
            "calibrated load levels (LP-referenced MLU targets)"]),
        "  { rank=same; teal; figret; zoo; }",
        "  }",
        # ---- twin layer
        '  subgraph cluster_twin { label="" style="rounded,filled" fillcolor="#F3FBF4" color="#3F8E4C" penwidth=1.6',
        header("h_twin", "VIRTUAL / DIGITAL-TWIN LAYER", "twin_h"),
        box("twin", "5-member physics-anchored\\nensemble twin", "twin", fontname=FONT_B),
        box("pred", "Predicted link loads\\nmean μ and uncertainty σ", "twin"),
        box("conf", "Conformal calibration\\n(validation only, c = 0.955)", "twin"),
        detail("d_twin", "Twin details", [
            "path features: demand, split fraction, hop count",
            "per-member MLP multiplier, initialised at physics (= 1)",
            "loss: MSE + anchor penalty (multiplier − 1)²",
            "μ = ensemble mean, σ = ensemble spread",
            "calibrated so 2σ coverage ≥ 0.9 on validation"]),
        "  { rank=same; twin; pred; conf; }",
        "  }",
        # ---- decision layer
        '  subgraph cluster_dec { label="" style="rounded,filled" fillcolor="#FFF8F0" color="#D27A1F" penwidth=1.6',
        header("h_dec", "SERVICE / DECISION LAYER", "dec_h"),
        box("policy", "Graph RL policy\\n(TopoFlowNet-style GNN)\\nsplits each demand over 4 paths", "dec", fontname=FONT_B),
        box("gate", "Safety gate\\naccept iff μ + kσ ≤ capacity\\non every link (k = 1)", "dec", fontname=FONT_B,
            penwidth="2.2"),
        box("accept", "Accept →\\napply proposal", "dec"),
        box("fallback", "Reject → fallback\\nlast accepted · shortest · equal\\n(conditional LP, optional)", "dec"),
        detail("d_dec", "Decision details", [
            "GNN on the link–path bipartite graph",
            "per-demand 4-way softmax split",
            "gate: utilisation-space UCB, limit 1.0",
            "optional relative-σ veto (off in frozen config)",
            "fallback: feasible first, else least-violating",
            "LP only if a forced-link lower bound ≤ 1"]),
        "  { rank=same; policy; gate; accept; }",
        "  }",
        box("net", "NETWORK  (flow-level simulator / physical proxy)\\napplies the routing and returns realised link loads",
            "net", fontname=FONT_B, width="9"),
        box("loop", "Closed-loop safe decision testing", "loop", fontname=FONT_B, shape="box",
            style="rounded,filled,bold", penwidth="2"),
        # invisible ordering
        '  h_data -> teal [style=invis]; h_twin -> twin [style=invis]; h_dec -> policy [style=invis];',
        '  d_data -> h_twin [style=invis weight=20]; d_twin -> h_dec [style=invis weight=20];',
        '  d_dec -> net [style=invis weight=20]; fallback -> net [style=invis];',
        '  prep -> d_data [style=invis]; twin -> d_twin [style=invis]; gate -> d_dec [style=invis];',
        '  h_data -> h_twin [style=invis weight=50]; h_twin -> h_dec [style=invis weight=50];',
        arrow("teal", "prep", "data"), arrow("figret", "prep", "data"), arrow("zoo", "prep", "data"),
        arrow("prep", "twin", "data", "topology, paths, TMs", constraint="false"),
        arrow("prep", "policy", "data", "bipartite graph, TM", constraint="false"),
        arrow("policy", "twin", "dec", "proposed action", constraint="false"),
        arrow("twin", "pred", "pred"), arrow("conf", "pred", "pred", "scale c"),
        arrow("pred", "gate", "pred", "μ, σ", constraint="false"),
        arrow("gate", "accept", "dec", "accept"), arrow("gate", "fallback", "dec", "reject"),
        arrow("accept", "net", "dec", "applied action"), arrow("fallback", "net", "dec", "applied action"),
        arrow("net", "twin", "fb", "realised loads / recalibration", constraint="false"),
        arrow("loop", "net", "fb", "", style="invis"),
        "}",
    ])


# ---------------------------------------------------------------- diagram 2
def d2_workflow() -> str:
    return "\n".join([
        f"digraph flow {{ {HEADER} rankdir=TB",
        '  label="Operational workflow: training (first run) and per-TM safe decision testing" labelloc=t fontsize=16 '
        'fontname="Helvetica-Bold"',
        box("start", "Start", "start", shape="box", style="rounded,filled,bold", fontname=FONT_B),
        box("load", "Load topology + traffic matrices", "data"),
        box("q1", "Trained policy\\nand twin exist?", "loop", shape="diamond", style="dashed,filled", fontname=FONT_B),
        box("pp", "Preprocess\\n(paths, bipartite graph, splits)", "data"),
        box("tt", "Train twin ensemble\\n(physics-anchored)", "twin"),
        box("cal", "Calibrate twin\\n(conformal, validation)", "twin"),
        box("tp", "Train policy (GNN)", "dec"),
        box("sel", "Select checkpoint\\non validation only", "dec"),
        box("lc", "Load checkpoints\\n(policy, twin, conformal scale)", "data"),
        box("loop_tm", "For each traffic matrix", "net", fontname=FONT_B),
        box("prop", "Policy proposes a split", "dec"),
        box("tw", "Twin predicts μ, σ\\nper link", "twin"),
        box("q2", "Gate accepts?\\nμ + kσ ≤ capacity", "dec_h", shape="diamond", style="dashed,filled", fontname=FONT_B,
            penwidth="2.2"),
        box("apply", "Apply proposal", "dec"),
        box("fb", "Fallback\\n(last accepted / shortest /\\nequal / conditional LP)", "dec_h", fontname=FONT_B),
        box("q3", "Failure or twin-error\\nscenario?", "loop", shape="diamond", style="dashed,filled", fontname=FONT_B),
        box("scen", "Inject scenario\\n(link failures E3 / twin error E4)", "eval"),
        box("ev", "Evaluate metrics\\nsatisfaction · MLU · unsafe accept · false reject", "eval", fontname=FONT_B),
        box("rep", "Results dashboard / reports", "eval"),
        box("end", "End", "start", shape="box", style="rounded,filled,bold", fontname=FONT_B),
        arrow("start", "load", "plain"), arrow("load", "q1", "plain"),
        arrow("q1", "pp", "plain", "No"), arrow("pp", "tt", "data"), arrow("tt", "cal", "pred"),
        arrow("cal", "tp", "dec"), arrow("tp", "sel", "dec"), arrow("sel", "loop_tm", "dec"),
        arrow("q1", "lc", "plain", "Yes"), arrow("lc", "loop_tm", "data"),
        arrow("loop_tm", "q3", "plain"), arrow("q3", "scen", "plain", "Yes"), arrow("q3", "prop", "plain", "No"),
        arrow("scen", "prop", "plain"),
        arrow("prop", "tw", "dec", "proposed action"), arrow("tw", "q2", "pred", "μ, σ"),
        arrow("q2", "apply", "dec", "Yes"), arrow("q2", "fb", "dec", "No"),
        arrow("apply", "ev", "dec"), arrow("fb", "ev", "dec"),
        arrow("ev", "loop_tm", "fb", "next TM", constraint="false"),
        arrow("ev", "rep", "plain", "all TMs done"), arrow("rep", "end", "plain"),
        "  { rank=same; pp; lc; }", "  { rank=same; apply; fb; }",
        "}",
    ])


# ---------------------------------------------------------------- diagram 3
def d3_experiments() -> str:
    return "\n".join([
        f"digraph exp {{ {HEADER} rankdir=LR splines=spline compound=true newrank=true",
        '  label="Experimental design" labelloc=t fontsize=16 fontname="Helvetica-Bold"',
        '  subgraph cluster_s { label="" style="rounded,filled" fillcolor="#F4F9FE" color="#3F75B0"',
        box("te", "Unseen test topologies\\nGEANT (FIGRET test) · Kdl", "data", fontname=FONT_B),
        box("ha", "Held aside (never used)\\nTM 28–35", "data", style="rounded,dashed,filled"),
        box("va", "Validation (tuning / selection only)\\nsame topologies, TM 20–27", "data"),
        box("tr", "Train topologies\\nB4 · UsCarrier · Cogentco\\nTM indices 0–19", "data"),
        header("h_s", "DATA SPLITS", "data_h"),
        "  }",
        '  subgraph cluster_e { label="" style="rounded,filled" fillcolor="#FBF8FE" color="#7B55A6"',
        box("cl", "Calibrated load\\nreference MLU = 0.5 / 0.7 / 0.9 / 1.1", "loop", fontname=FONT_B),
        box("e4", "E4  twin-error levels\\nmember-wise σ, degraded twins", "eval"),
        box("e3", "E3  link failures\\n0–3 failed links, 10 seeds", "eval"),
        box("e2", "E2  unseen topologies\\nholdout A–F on GEANT, Kdl", "eval"),
        box("e1", "E1  seen topologies\\n(P1 setting; B4 baseline freeze)", "eval"),
        header("h_e", "EXPERIMENTS", "eval_h"),
        "  }",
        '  subgraph cluster_b { label="" style="rounded,filled" fillcolor="#FFF8F0" color="#D27A1F"',
        detail("ab", "Ablations", ["(a) no gate", "(b) gate without uncertainty", "(c) no fallback",
                                   "(d) single-topology (B4-only) policy", "(e) twin without learned residual",
                                   "Dm: mean-only gate with tuned margin"]),
        detail("bl", "Baselines", ["shortest path, ECMP / equal split", "LP (min-MLU; max-throughput for P1)",
                                   "frozen P1 GRL-TE (B4, GEANT)", "policy without twin or gate (A)",
                                   "gate without uncertainty (D)", "Teal, FIGRET, NCFlow: quoted from P1"]),
        header("h_b", "BASELINES & ABLATIONS", "dec_h"),
        "  }",
        '  subgraph cluster_m { label="" style="rounded,filled" fillcolor="#F3FBF4" color="#3F8E4C"',
        detail("st", "Statistics", ["mean ± SD over seeds", "paired Wilcoxon, Holm correction, Cliff's δ",
                                    "McNemar for paired accept/reject", "Friedman + Nemenyi across topologies"]),
        detail("mt", "Metrics", ["demand satisfaction (P1)", "max link utilisation", "violation rate",
                                 "unsafe-accept, false-reject rates", "generalization gap (seen − unseen)",
                                 "decision time per component"]),
        header("h_m", "METRICS & STATISTICS", "twin_h"),
        "  }",
        # one arrow between consecutive stages (cluster to cluster), plus the vertical stacking inside each stage
        arrow("va", "e2", "data", "", ltail="cluster_s", lhead="cluster_e"),
        arrow("e2", "ab", "dec", "", ltail="cluster_e", lhead="cluster_b"),
        arrow("ab", "st", "pred", "", ltail="cluster_b", lhead="cluster_m"),
        '  { rank=same; h_s; tr; va; ha; te; }',
        '  { rank=same; h_e; e1; e2; e3; e4; cl; }',
        '  { rank=same; h_b; bl; ab; }',
        '  { rank=same; h_m; mt; st; }',
        "}",
    ])


# ---------------------------------------------------------------- diagram 4
def d4_gate() -> str:
    return "\n".join([
        f"digraph gate {{ {HEADER} rankdir=TB",
        '  label="Safety-gate decision logic and fallback order" labelloc=t fontsize=16 fontname="Helvetica-Bold"',
        box("in", "Proposed split a\\n(from graph RL policy)", "dec"),
        box("tw", "Twin ensemble (5 members)\\nlink loads l_1(e) ... l_5(e)", "twin"),
        box("mu", "μ(e) = mean_j l_j(e)\\nσ(e) = c · std_j l_j(e)\\nc = conformal scale (validation, 0.955)", "twin"),
        box("util", "Utilisation space\\nu(e) = μ(e) / cap(e),   s(e) = σ(e) / cap(e)", "twin"),
        box("q0", "All predictions\\nfinite?", "loop", shape="diamond", style="dashed,filled", fontname=FONT_B),
        box("q1", "Relative-σ check\\ns(e) ≤ s_max for all e ?\\n(optional; off in frozen config)", "loop",
            shape="diamond", style="dashed,filled"),
        box("q2", "Capacity check\\nu(e) + k · s(e) ≤ 1 for all links e\\n(k = 1;  k = 0 gives mean-only gate D)",
            "dec_h", shape="diamond", style="dashed,filled", fontname=FONT_B, penwidth="2.2"),
        box("acc", "ACCEPT\\napply proposal", "start", fontname=FONT_B),
        box("rej", "REJECT\\n(record reason)", "dec_h", fontname=FONT_B),
        box("f1", "1. last accepted routing, if still feasible\\n(sequential controller only; not used in\\n"
                  "the frozen paired A–F evaluation)", "dec"),
        box("f2", "2. shortest path / equal split\\nfeasible first, else least-violating", "dec"),
        box("q3", "Forced-link lower bound\\n≤ 1 ?", "loop", shape="diamond", style="dashed,filled"),
        box("lp", "3. Conditional LP\\n(min-MLU over k = 4 paths, HiGHS)\\napplied only if verified feasible", "dec"),
        box("keep", "keep step-2 fallback\\n(no feasible route exists)", "net"),
        box("net", "Applied routing → network", "net", fontname=FONT_B),
        arrow("in", "tw", "dec", "proposed action"), arrow("tw", "mu", "pred"), arrow("mu", "util", "pred"),
        arrow("util", "q0", "pred"), arrow("q0", "q1", "plain", "yes"), arrow("q0", "rej", "plain", "no"),
        arrow("q1", "q2", "plain", "yes"), arrow("q1", "rej", "plain", "no"),
        arrow("q2", "acc", "dec", "yes"), arrow("q2", "rej", "dec", "no"),
        arrow("rej", "f1", "dec"), arrow("f1", "f2", "dec", "infeasible"), arrow("f2", "q3", "dec", "still overloads"),
        arrow("q3", "lp", "dec", "yes"), arrow("q3", "keep", "dec", "no"),
        arrow("acc", "net", "dec"), arrow("f1", "net", "dec", "feasible"), arrow("f2", "net", "dec", "feasible"),
        arrow("lp", "net", "dec"), arrow("keep", "net", "dec"),
        "  { rank=same; acc; rej; }",
        "}",
    ])


# ---------------------------------------------------------------- drawio conversion
def _style(attrs: dict) -> str:
    shape = attrs.get("shape", "box")
    st = attrs.get("style", "")
    parts = ["whiteSpace=wrap", "html=1", f"fontFamily={FONT}"]
    parts.append("rhombus" if shape == "diamond" else "rounded=1")
    parts.append(f"fillColor={attrs.get('fillcolor', '#FFFFFF')}")
    parts.append(f"strokeColor={attrs.get('color', '#000000')}")
    parts.append(f"strokeWidth={attrs.get('penwidth', '1.4')}")
    parts.append(f"fontSize={attrs.get('fontsize', '12')}")
    if "dashed" in st:
        parts.append("dashed=1")
    if "Bold" in attrs.get("fontname", ""):
        parts.append("fontStyle=1")
    if "\\l" in attrs.get("label", ""):
        parts.append("align=left;spacingLeft=8")
    return ";".join(parts) + ";"


def _label(raw: str) -> str:
    text = raw.replace("\\l", "\n").replace("\\n", "\n").rstrip("\n")
    return "<br>".join(html.escape(line) for line in text.split("\n"))


def to_drawio(name: str, layout: dict) -> str:
    h = float(layout["bb"].split(",")[3])
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    gid = {}
    k = 2
    for obj in layout.get("objects", []):
        if obj.get("bb") and obj["name"].startswith("cluster"):  # layer background
            x0, y0, x1, y1 = (float(v) for v in obj["bb"].split(","))
            style = (f"rounded=1;arcSize=4;fillColor={obj.get('fillcolor', '#FFFFFF')};"
                     f"strokeColor={obj.get('color', '#999999')};strokeWidth=1.6;")
            cells.append(f'<mxCell id="{k}" value="" style="{style}" vertex="1" parent="1">'
                         f'<mxGeometry x="{x0:.1f}" y="{h - y1:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
                         f'as="geometry"/></mxCell>')
            k += 1
    for obj in layout.get("objects", []):
        if "pos" not in obj:
            continue
        x, y = (float(v) for v in obj["pos"].split(","))
        w, hh = float(obj["width"]) * 72, float(obj["height"]) * 72
        gid[obj["_gvid"]] = k
        cells.append(f'<mxCell id="{k}" value="{escape(_label(obj.get("label", "")), {chr(34): "&quot;"})}" '
                     f'style="{_style(obj)}" vertex="1" parent="1"><mxGeometry x="{x - w / 2:.1f}" y="{h - y - hh / 2:.1f}" '
                     f'width="{w:.1f}" height="{hh:.1f}" as="geometry"/></mxCell>')
        k += 1
    for e in layout.get("edges", []):
        if e.get("style") == "invis" or e["tail"] not in gid or e["head"] not in gid:
            continue
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={e.get('color', '#555555')};"
                 f"fontColor={e.get('fontcolor', '#555555')};strokeWidth=1.6;endArrow=block;endFill=1;fontSize=10;"
                 f"fontFamily={FONT};" + ("dashed=1;" if e.get("style") == "dashed" else ""))
        lab = escape(_label(e.get("xlabel", "").strip()), {chr(34): "&quot;"})
        cells.append(f'<mxCell id="{k}" value="{lab}" style="{style}" edge="1" parent="1" source="{gid[e["tail"]]}" '
                     f'target="{gid[e["head"]]}"><mxGeometry relative="1" as="geometry"/></mxCell>')
        k += 1
    body = "".join(cells)
    return (f'<mxfile host="make_diagrams.py"><diagram name="{name}"><mxGraphModel dx="1200" dy="900" grid="1" '
            f'gridSize="10" guides="1" page="1" pageScale="1" background="#FFFFFF"><root>{body}</root>'
            f'</mxGraphModel></diagram></mxfile>\n')


def render(name: str, src: str) -> None:
    dot_path = HERE / f"{name}.dot"
    dot_path.write_text(src, encoding="utf-8")
    for fmt, extra in (("svg", []), ("pdf", []), ("png", ["-Gdpi=300"])):
        subprocess.run([DOT, f"-T{fmt}", *extra, str(dot_path), "-o", str(HERE / f"{name}.{fmt}")], check=True)
    layout = json.loads(subprocess.run([DOT, "-Tjson", str(dot_path)], check=True, capture_output=True,
                                       text=True, encoding="utf-8").stdout)
    (HERE / f"{name}.drawio").write_text(to_drawio(name, layout), encoding="utf-8")
    print(f"{name}: dot, svg, pdf, png (300 dpi), drawio")


def main() -> None:
    for name, fn in (("fig1_three_layer_architecture", d1_architecture), ("fig2_operational_workflow", d2_workflow),
                     ("fig3_experimental_design", d3_experiments), ("fig4_safety_gate_logic", d4_gate)):
        render(name, fn())


if __name__ == "__main__":
    main()
