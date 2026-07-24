#!/usr/bin/env python3
"""
Measure tikzposter block extents by probing the class's own vertical cursor.

Rather than estimating overflow from word counts, this compiles the poster on an
oversized page (so nothing is clipped) and reads tikzposter's internal
\\TP@blocktop dimen after every top-level \\block. That dimen is the running
placement cursor, set globally by the class at tikzposter.cls:430-433.

Geometry note: the tikzpicture is centred on the page, so y=0 is the page
centre and the content top sits at +paperheight/2. Growing the page therefore
shifts every absolute coordinate. We avoid depending on that shift by working
only in *relative* terms:

    used      = coltop - colbottom          (measured on the tall page)
    available = coltop_real + paperheight_real/2   (measured on the real page)
    overflow  = used - available

Usage:
    python tools/measure_poster.py base_poster.tex --tall 64
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

MIKTEX = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64"
PT_PER_IN = 72.27  # TeX pt

PROBE_PREAMBLE = r"""
\makeatletter
\newcommand{\POSPROBE}[1]{%
  \typeout{@@POS@@ #1 @@ \the\TP@blocktop @@ \the\TP@colbottom @@ \the\TP@subcolbottom}%
}
\makeatother
"""


def find_matching(s, i):
    """Given s[i] == '{', return index of the matching '}'."""
    assert s[i] == "{", s[i : i + 20]
    depth = 0
    while i < len(s):
        c = s[i]
        if c == "\\":  # skip escaped char
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces")


def instrument(src):
    """Inject \\POSPROBE after every top-level \\block and at each column start."""
    out = []
    pos = 0
    n = 0
    for m in re.finditer(r"\\block(?=\[|\{)", src):
        if m.start() < pos:
            continue
        i = m.end()
        if src[i] == "[":  # optional arg
            i = src.index("]", i) + 1
        j = find_matching(src, i)  # title
        title = src[i + 1 : j]
        k = find_matching(src, j + 1)  # body
        n += 1
        out.append(src[pos : k + 1])
        out.append("\n\\POSPROBE{BLOCK %d %s}\n" % (n, title.replace("}", "").replace("{", "")))
        pos = k + 1
    out.append(src[pos:])
    s = "".join(out)

    # Probe column / subcolumn structure boundaries too.
    s = re.sub(r"(\\column\{([\d.]+)\})", r"\1\\POSPROBE{COLSTART \2}", s)
    s = re.sub(r"(\\subcolumn\{([\d.]+)\})", r"\1\\POSPROBE{SUBCOLSTART \2}", s)
    s = s.replace(r"\begin{subcolumns}", r"\POSPROBE{SUBCOLS-BEGIN}\begin{subcolumns}")
    s = s.replace(r"\end{subcolumns}", r"\end{subcolumns}\POSPROBE{SUBCOLS-END}")
    s = s.replace(r"\end{columns}", r"\POSPROBE{COLS-END}\end{columns}")
    s = s.replace(r"\begin{columns}", r"\begin{columns}\POSPROBE{COLS-BEGIN}")
    s = s.replace(r"\begin{document}", PROBE_PREAMBLE + r"\begin{document}")
    return s


def set_height(src, inches):
    src = re.sub(r"paperheight=[\d.]+in", "paperheight=%gin" % inches, src)
    return src


def compile_and_probe(src, workdir, jobname, repo):
    workdir.mkdir(parents=True, exist_ok=True)
    tex = workdir / (jobname + ".tex")
    tex.write_text(src, encoding="latin-1")
    env_exe = str(MIKTEX / "pdflatex.exe")
    cmd = [
        env_exe,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "--max-print-line=2000",
        "-output-directory=" + str(workdir),
        "-jobname=" + jobname,
        str(tex),
    ]
    # Run from the repo root so \includegraphics{figures/...} resolves.
    r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, errors="replace")
    log = (workdir / (jobname + ".log")).read_text(encoding="latin-1", errors="replace")
    if r.returncode != 0:
        errs = [l for l in log.splitlines() if l.startswith("!")]
        print("BUILD FAILED (%s). First errors:" % jobname, file=sys.stderr)
        for e in errs[:15]:
            print("   ", e, file=sys.stderr)
        raise SystemExit(1)
    probes = []
    for line in log.splitlines():
        if line.startswith("@@POS@@"):
            parts = [p.strip() for p in line[len("@@POS@@") :].split("@@")]
            name = parts[0]
            vals = [float(v.replace("pt", "")) / PT_PER_IN for v in parts[1:4]]
            probes.append((name, *vals))
    return probes, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--tall", type=float, default=64.0)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    repo = Path(args.tex).resolve().parent
    src = Path(args.tex).read_text(encoding="latin-1")
    m = re.search(r"paperheight=([\d.]+)in", src)
    real_h = float(m.group(1))
    work = Path(args.workdir) if args.workdir else repo / "_measure"

    inst = instrument(src)
    tall_probes, _ = compile_and_probe(set_height(inst, args.tall), work, "measure_tall", repo)
    real_probes, _ = compile_and_probe(inst, work, "measure_real", repo)

    print("=" * 78)
    print("BLOCK EXTENTS  (tall page = %gin, real page = %gin)" % (args.tall, real_h))
    print("=" * 78)

    # coltop on the real page: cursor at \begin{columns}
    coltop_real = [p for p in real_probes if p[0] == "COLS-BEGIN"][0][1]
    available = coltop_real + real_h / 2.0
    print("content top (real page)      : %+8.3f in  (y, page centre = 0)" % coltop_real)
    print("page bottom edge (real page) : %+8.3f in" % (-real_h / 2.0))
    print("AVAILABLE COLUMN HEIGHT      : %8.3f in" % available)
    print()

    # Walk the tall-page probes, tracking per-column consumption.
    cur_top = None
    prev = None
    col_label = None
    print("%-46s %9s %9s" % ("probe", "cursor_y", "block_h"))
    print("-" * 78)
    for name, blocktop, colbot, subcolbot in tall_probes:
        h = ""
        if name.startswith("BLOCK") and prev is not None:
            h = "%9.3f" % (prev - blocktop)
        if name.startswith(("COLSTART", "SUBCOLSTART", "COLS-BEGIN", "SUBCOLS-BEGIN")):
            cur_top = blocktop
        print("%-46s %+9.3f %9s" % (name[:46], blocktop, h))
        prev = blocktop
    print()

    # Per-column totals: recompute by replaying the structure.
    print("=" * 78)
    print("COLUMN BUDGETS")
    print("=" * 78)
    tops = {}
    order = []
    cur = None
    top_stack = []
    colbegin = [p for p in tall_probes if p[0] == "COLS-BEGIN"][0][1]
    for name, blocktop, colbot, subcolbot in tall_probes:
        if name.startswith("COLSTART"):
            cur = "column " + name.split()[1]
            tops[cur] = colbegin
            order.append(cur)
        elif name.startswith("SUBCOLSTART"):
            cur = "  subcol " + name.split()[1]
            tops[cur] = top_stack[-1] if top_stack else colbegin
            order.append(cur)
        elif name == "SUBCOLS-BEGIN":
            top_stack.append(blocktop)
        elif name == "SUBCOLS-END":
            if top_stack:
                top_stack.pop()
            cur = order[-3] if len(order) >= 3 else cur  # back to outer column
        if cur is not None:
            tops.setdefault(cur, blocktop)
        if name.startswith("BLOCK") and cur is not None:
            tops[cur + "__bot"] = blocktop

    for key in order:
        bot = tops.get(key + "__bot")
        if bot is None:
            continue
        used = tops[key] - bot
        slack = available - used
        flag = "OVER by %6.2f in" % (-slack) if slack < 0 else "slack %6.2f in" % slack
        print("%-24s used %7.3f in   %s" % (key, used, flag))
    print()
    print("(subcolumn budgets are measured against the full column height;")
    print(" the References block below the subcolumns is charged to the column.)")


if __name__ == "__main__":
    main()
