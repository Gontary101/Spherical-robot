"""
Manufacturing outputs from the build CAD -> cad/build/out/release/

    gyra_mk21_build.step        full assembly (every part named + coloured, fasteners included)
    parts/<drawing>.step        one solid per custom drawing (left-hand / first instance)
    stl/<drawing>.stl           printed parts (MJF PA12) and the PC parts
    dxf/<drawing>.dxf           flat profile of every plate part (waterjet / laser), holes included
    drawings/<drawing>.pdf      drawing sheet: 3 orthographic views + iso (hidden-line removed), title block
                                (material, process, finish, mass, qty), overall size and the hole table from the joints
    drawings/GYRA_Mk21_drawing_set.pdf   all sheets

    python cad/build/export.py [--no-step] [--only NAME]
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import cadquery as cq  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Curve  # noqa: E402
from OCP.GCPnts import GCPnts_QuasiUniformDeflection  # noqa: E402
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt  # noqa: E402
from OCP.HLRAlgo import HLRAlgo_Projector  # noqa: E402
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

from assemble import OUT, build_all  # noqa: E402
from bom import base_name  # noqa: E402
from lib import mass_props  # noqa: E402

REL = os.path.join(OUT, "release")
PRINTED = ("pod_shield", "cam_holder", "light_ring", "cam_window")
PLATES = ("hanger", "arc_plate", "carriage", "outer_plate", "pod_plate", "lidar_riser", "arc_bracket", "gnss_plate", "track")


def hlr_edges(shape, view_dir, up):
    """Visible and hidden 2-D polylines of `shape` seen along -view_dir with `up` vertical."""
    algo = HLRBRep_Algo()
    algo.Add(shape.wrapped)
    n, u = cq.Vector(*view_dir).normalized(), cq.Vector(*up)
    vx = u.cross(n)                                   # screen x, so that screen y = n x vx = up
    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(n.x, n.y, n.z), gp_Dir(vx.x, vx.y, vx.z))
    algo.Projector(HLRAlgo_Projector(ax))
    algo.Update()
    algo.Hide()
    h = HLRBRep_HLRToShape(algo)
    out = {}
    for key, comp in (("vis", h.VCompound()), ("vis", h.OutLineVCompound()), ("hid", h.HCompound())):
        if comp is None or comp.IsNull():
            continue
        exp = TopExp_Explorer(comp, TopAbs_EDGE)
        while exp.More():
            e = TopoDS.Edge_s(exp.Current())
            c = BRepAdaptor_Curve(e)
            d = GCPnts_QuasiUniformDeflection(c, 0.05)
            if d.IsDone() and d.NbPoints() > 1:
                pts = [d.Value(i) for i in range(1, d.NbPoints() + 1)]
                out.setdefault(key, []).append([(p.X(), p.Y()) for p in pts])
            exp.Next()
    return out


VIEWS = {"front (-y)": ((0, -1, 0), (0, 0, 1)), "top (+z)": ((0, 0, 1), (0, 1, 0)), "side (+x)": ((1, 0, 0), (0, 0, 1)),
         "iso": ((1, -1, 1), (0, 0, 1))}


def principal_frame(shape):
    """Recentre a part at its bounding-box centre (drawings show it about its own origin)."""
    bb = shape.BoundingBox()
    c = cq.Vector((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
    return shape.translate(-c), c


def drawing_sheet(pdf, name, part, qty, joints_by_member):
    shape, c0 = principal_frame(part.shape)
    bb = shape.BoundingBox()
    fig = plt.figure(figsize=(16.54, 11.69))                    # A3 landscape
    fig.patch.set_facecolor("white")
    # the view with the largest projected area gets the big frame
    area = {"front (-y)": bb.xlen * bb.zlen, "top (+z)": bb.xlen * bb.ylen, "side (+x)": bb.ylen * bb.zlen}
    order = sorted(area, key=area.get, reverse=True) + ["iso"]
    frames = [[0.03, 0.31, 0.55, 0.64], [0.61, 0.66, 0.36, 0.29], [0.61, 0.31, 0.36, 0.31], [0.03, 0.03, 0.24, 0.25]]
    for vn, fr in zip(order, frames):
        vd, up = VIEWS[vn]
        ax = fig.add_axes(fr)
        try:
            E = hlr_edges(shape, vd, up)
        except Exception:
            E = {}
        for pl in E.get("hid", []):
            xs, ys = zip(*pl)
            ax.plot(xs, ys, color="#9aa3b5", lw=0.4, ls=(0, (3, 2)))
        for pl in E.get("vis", []):
            xs, ys = zip(*pl)
            ax.plot(xs, ys, color="#111827", lw=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(vn + ("" if vn == "iso" else "   (axes in mm from the view origin)"), fontsize=8.5, loc="left")
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.2, color="#e5e7eb")
    sp = part.spec
    mp = mass_props(part)
    lines = [f"DRAWING  {name}", f"qty per robot: {qty}" + ("  (R-hand part = this one mirrored about XZ)" if qty > 1 else ""),
             f"material: {sp.get('material', '')}", f"process: {sp.get('process', '')}", f"finish: {sp.get('finish', '')}",
             f"mass: {mp['mass'] * 1000:.1f} g  overall: {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm (x, y, z)",
             f"note: {sp.get('note', sp.get('fit', sp.get('gear', '')))}"[:80],
             "general tolerances ISO 2768-mK, edges broken 0.3 x 45, units mm",
             "geometry master: parts/" + name + ".step",
             f"view origin = bounding-box centre = robot ({c0.x:.2f}, {c0.y:.2f}, {c0.z:.2f})"]
    import textwrap
    lines = [w for ln in lines for w in textwrap.wrap(ln, 82, subsequent_indent="    ")]
    fig.text(0.29, 0.27, "\n".join(lines), fontsize=7.5, family="DejaVu Sans", va="top")
    holes = joints_by_member.get(part.name, [])
    if holes:
        rows = ["HOLE TABLE (view coords of the screw seat, mm)", f"{'feature':34s} {'x':>7s} {'y':>7s} {'z':>7s}  axis"]
        for j, feat in sorted(holes, key=lambda h: (h[1], h[0]["seat"])):
            s = j["seat"]
            n = j["n"]
            rows.append(f"{feat[:34]:34s} {s[0] - c0.x:7.2f} {s[1] - c0.y:7.2f} {s[2] - c0.z:7.2f}  "
                        f"{'+-'[n[0] < 0] if abs(n[0]) > .9 else ''}{'x' if abs(n[0]) > .9 else ''}"
                        f"{'+-'[n[1] < 0] if abs(n[1]) > .9 else ''}{'y' if abs(n[1]) > .9 else ''}"
                        f"{'+-'[n[2] < 0] if abs(n[2]) > .9 else ''}{'z' if abs(n[2]) > .9 else ''}"
                        f"{'' if max(abs(c) for c in n) > .9 else f'({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})'}")
        col = 26
        chunks = [rows[:2] + rows[2 + i:2 + i + col] for i in range(0, len(rows) - 2, col)][:2]
        txt = chunks[0]
        if len(rows) - 2 > col:
            txt = txt + [f"... {len(rows) - 2 - col} more holes: see joints.json / parts STEP"] if len(rows) - 2 > 2 * col else txt
        fig.text(0.575, 0.27, "\n".join(txt), fontsize=5.6, family="DejaVu Sans Mono", va="top")
        if len(chunks) > 1 and len(rows) - 2 <= 2 * col:
            fig.text(0.78, 0.27, "\n".join(chunks[1]), fontsize=5.6, family="DejaVu Sans Mono", va="top")
    fig.text(0.97, 0.012, "GYRA Mk2.1 build  -  generated from cad/build (do not scale)", fontsize=7, ha="right", color="#6b7280")
    pdf.savefig(fig)
    if os.environ.get("GYRA_SINGLE_PDF", "1") == "1":
        with PdfPages(os.path.join(REL, "drawings", f"{name}.pdf")) as one:
            one.savefig(fig)
    plt.close(fig)


def hole_features(joints):
    """member part -> [(joint dict, feature text)]: what each joint machines into each member."""
    from lib import CBORE, CLEAR, INSERT, TAPDRILL
    by = {}
    for j in joints:
        d = j["d"]
        mem = j["members"]
        end = j["end"]
        for k, m in enumerate(mem):
            last = k == len(mem) - 1
            if end == "tap" and last:
                f = f"M{d:g} tapped {j.get('tapped_depth', 0):.1f} deep (drill {TAPDRILL[d]})"
            elif end == "insert" and last:
                f = f"d{INSERT[d][0]} x {INSERT[d][1] + 1:.1f} for M{d:g} heat-set insert"
            elif end == "pem" and last:
                f = f"d{d + 1.5:.1f} for M{d:g} PEM nut"
            else:
                f = f"d{CLEAR[d]} through" + (f", cbore d{CBORE[d][0]}" if k == 0 and j.get("head") == "SHCS" and j.get("grip", 0) < 0 else "")
                if k == 0 and j.get("head") == "FHCS":
                    f = f"d{CLEAR[d]} through, csk 90 deg"
            by.setdefault(m, []).append((j, f))
    return by


def plate_dxf(part, path):
    """Profile of a plate at its mid-thickness (section normal = thinnest bounding-box axis)."""
    import ezdxf
    bb = part.shape.BoundingBox()
    dims = {"x": bb.xlen, "y": bb.ylen, "z": bb.zlen}
    ax = min(dims, key=dims.get)
    mid = {"x": (bb.xmin + bb.xmax) / 2, "y": (bb.ymin + bb.ymax) / 2, "z": (bb.zmin + bb.zmax) / 2}[ax]
    normal = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax]
    pl = cq.Plane(origin=tuple(c * mid for c in normal) if False else {"x": (mid, 0, 0), "y": (0, mid, 0), "z": (0, 0, mid)}[ax],
                  xDir={"x": (0, 1, 0), "y": (1, 0, 0), "z": (1, 0, 0)}[ax], normal=normal)
    sec = part.shape.intersect(cq.Workplane(pl).rect(4000, 4000).extrude(0.01).val())
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for e in sec.Edges():
        pts = [pl.toLocalCoords(v) for v in e.positions([i / 24 for i in range(25)])] if e.geomType() != "LINE" else \
            [pl.toLocalCoords(e.startPoint()), pl.toLocalCoords(e.endPoint())]
        msp.add_lwpolyline([(p.x, p.y) for p in pts])
    doc.header["$INSUNITS"] = 4
    doc.saveas(path)
    return dims[ax]


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for d in ("parts", "stl", "dxf", "drawings"):
        os.makedirs(os.path.join(REL, d), exist_ok=True)
    parts, joints, threads, bonds = build_all(verbose=False)
    jinfo = [dict(name=j.name, **j.info) for j in joints]
    holes = hole_features(jinfo)
    # one representative per drawing
    groups = {}
    for p in parts:
        if p.kind != "custom":
            continue
        groups.setdefault(base_name(p.name), []).append(p)
    if "--meshes" in sys.argv:                      # per-part STL + manifest for render/render_build.py
        md = os.path.join(REL, "mesh")
        os.makedirs(md, exist_ok=True)
        man = []
        for p in parts:
            fn = os.path.join(md, f"{p.name}.stl")
            p.shape.exportStl(fn, tolerance=0.08 if p.kind == "fastener" else 0.15, angularTolerance=0.3)
            man.append(dict(name=p.name, body=p.body, render=p.render, kind=p.kind, file=f"{p.name}.stl"))
        json.dump(man, open(os.path.join(md, "manifest.json"), "w"), indent=0)
        print(f"{len(man)} meshes -> {md}", flush=True)
        if "--meshes-only" in sys.argv:
            return
    if "--no-step" not in sys.argv and not only:
        from render_colors import RGB
        asm = cq.Assembly(name="GYRA_Mk21_build")
        for p in parts:
            r, g, b = RGB.get(p.render, (0.6, 0.6, 0.6))
            asm.add(p.shape, name=p.name, color=cq.Color(r, g, b))
        asm.save(os.path.join(REL, "gyra_mk21_build.step"), exportType="STEP")
        print("assembly STEP written", flush=True)
    with PdfPages(os.path.join(REL, "drawings", "GYRA_Mk21_drawing_set.pdf")) as pdf:
        for name, ps in groups.items():
            if only and only != name:
                continue
            p = ps[0]
            if not only:
                cq.exporters.export(p.shape, os.path.join(REL, "parts", f"{name}.step"))
            if name.startswith(PRINTED):
                p.shape.exportStl(os.path.join(REL, "stl", f"{name}.stl"), tolerance=0.05, angularTolerance=0.1)
            if name.startswith(PLATES):
                try:
                    t = plate_dxf(p, os.path.join(REL, "dxf", f"{name}.dxf"))
                except Exception as e:  # curved parts that are not true plates
                    print(f"  dxf skipped for {name}: {e}")
            drawing_sheet(pdf, name, p, len(ps), holes)
            print(f"  {name}: {len(ps)} x", flush=True)
    print(f"release written to {REL}")


if __name__ == "__main__":
    main()
