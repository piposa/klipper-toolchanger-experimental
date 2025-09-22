#!/usr/bin/env python3
"""
Post‑processor for multimaterial g‑code.

1. Rewrites every bare `Tn` line to `Tn X=… Y=… Z=…`, using the first G0/G1
   after it that carries both X and Y (extrusion not required).

2. As soon as a tool is left *for the final time*, injects
   `M104 S0 Tn ; auto-off unused` after the *next* tool‑change so the old
   nozzle cools while printing continues.
   • Single‑tool jobs are untouched (no stray M104).
   • The very last active tool is **not** cooled at EOF.

3. Retains the original dangling‑tower prune logic.

Usage:
    python3 restore_pos_fix.py /path/to/file.gcode
"""

import sys, pathlib, re, traceback

# ── regular expressions ──────────────────────────────────────────
pat_T  = re.compile(r'^\s*(T(\d+))\s*$', re.I)
pat_xy = re.compile(r'G1\b[^;]*?\bX([-+]?\d*\.?\d+)[^;]*?\bY([-+]?\d*\.?\d+)', re.I)
pat_z  = re.compile(r'\bZ([-+]?\d*\.?\d+)', re.I)
xy_cmd = re.compile(r'^[GM]\d+[^;]*\bX[-+]?\d')

# ─────────────────────────────────────────────────────────────────
def patch_file(path_str: str) -> None:
    p     = pathlib.Path(path_str)
    lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()

    # --- pass 0: where does each tool last appear? ---
    last_use = {}
    for idx, ln in enumerate(lines):
        m = pat_T.match(ln)
        if m:
            last_use[int(m.group(2))] = idx

    # --- helper: pick coords for a tool‑change ---
    def find_target(start: int):
        """Return (X, Y, Z) of the first G‑move with both X and Y after *start*."""
        for j in range(start, len(lines)):
            ln = lines[j]
            if ln.startswith(';'):
                continue
            if pat_T.match(ln) and j != start:   # next tool‑change → stop
                break
            if ln.startswith(('G0', 'G1')):
                mxy = pat_xy.search(ln)
                if mxy:
                    x = float(mxy.group(1))
                    y = float(mxy.group(2))
                    mz = pat_z.search(ln)
                    z = float(mz.group(1)) if mz else None
                    return x, y, z
        return (None, None, None)

    out           = []
    last_z        = 0.2
    pending_cool  = None    # tool to cool immediately after next T
    i             = 0

    # --- main rewrite ---
    while i < len(lines):
        ln = lines[i]

        # track current Z
        if ln.startswith(('G0', 'G1')):
            mz = pat_z.search(ln)
            if mz:
                last_z = float(mz.group(1))

        mT = pat_T.match(ln)
        if mT:
            tool = int(mT.group(2))

            # 1) patch coordinates
            x, y, z = find_target(i + 1)
            if x is not None:
                if z is None:
                    z = last_z
                ln = f'{mT.group(1)} X={x:.3f} Y={y:.3f} Z={z:.3f}'
            out.append(ln)

            # 2) cool previous tool if queued
            if pending_cool is not None:
                out.append(f'M104 S0 T{pending_cool} ; auto-off unused')
                pending_cool = None

            # 3) queue current tool if this is its last use
            if i == last_use[tool]:
                pending_cool = tool

            i += 1
            continue

        out.append(ln)
        i += 1

    # --- dangling prime‑tower prune (unchanged) ---
    def prune_last_tower(raw):
        end_excl = None
        for idx, ln in enumerate(raw):
            if ln.startswith('EXCLUDE_OBJECT_END'):
                end_excl = idx
        if end_excl is None:
            return raw
        start = stop = None
        for j in range(end_excl + 1, len(raw)):
            if ';TYPE:Prime tower' in raw[j]:
                start = j
                break
        if start is None:
            return raw
        for k in range(start, len(raw)):
            if raw[k].startswith('; CP TOOLCHANGE END'):
                stop = k
                break
        if stop is None:
            return raw
        for m in range(stop + 1, len(raw)):
            if xy_cmd.search(raw[m]):
                return raw
        return raw[:end_excl + 1] + raw[stop + 1:]

    lines = prune_last_tower(out)
    lines.append('; === PATCHED ===')
    p.write_text('\n'.join(lines), encoding='utf-8')

# --- CLI ---------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(0)
    try:
        patch_file(sys.argv[1])
    except Exception:
        pathlib.Path(sys.argv[1] + '.err').write_text(
            traceback.format_exc(), encoding='utf-8'
        )
        sys.exit(1)
