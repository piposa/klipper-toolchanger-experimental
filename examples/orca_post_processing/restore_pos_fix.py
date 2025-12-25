#!/usr/bin/env python3
"""
Post-processor for multimaterial g-code.

1. Rewrites every bare `Tn` line to `Tn X=… Y=… Z=…`, using the *last* G0/G1
   travel before the first extrusion after it:
   - X/Y from the last move that has both X and Y
   - Z from the last Z seen in that region (including Z-only moves)

2. As soon as a tool is left *for the final time*, injects
   `M104 S0 Tn ; auto-off unused` after the *next* tool-change so the old
   nozzle cools while printing continues.
   • Single-tool jobs are untouched (no stray M104).
   • The very last active tool is **not** cooled at EOF.
"""

import sys, pathlib, re, traceback

# ── regular expressions ──────────────────────────────────────────
pat_T  = re.compile(r'^\s*(T(\d+))\s*$', re.I)
pat_xy = re.compile(r'G1\b[^;]*?\bX([-+]?\d*\.?\d+)[^;]*?\bY([-+]?\d*\.?\d+)', re.I)
pat_z  = re.compile(r'\bZ([-+]?\d*\.?\d+)', re.I)
pat_e  = re.compile(r'\bE([-+]?\d*\.?\d+)', re.I)
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

    # --- helper: pick coords for a tool-change ---
    def find_target(start: int, fallback_z: float):
        """
        Return (X, Y, Z) for a tool-change at line index `start`.

        Strategy:
        • Scan forward until:
          - the next tool-change, OR
          - the first line with an E move (extrusion), whichever comes first.
        • Within that window, track:
          - last X/Y from any G0/G1 with both X and Y
          - last Z from any Z in those moves (including Z-only moves)
        • If no suitable X/Y is found, return (None, None, None).
        """
        last_x = None
        last_y = None
        last_z_local = None

        for j in range(start, len(lines)):
            ln = lines[j]
            if ln.startswith(';'):
                continue

            # stop at next toolchange (but not the one at `start` itself)
            if pat_T.match(ln) and j != start:
                break

            if ln.startswith(('G0', 'G1')):
                # if this has extrusion, stop BEFORE using this line
                if pat_e.search(ln):
                    break

                mxy = pat_xy.search(ln)
                if mxy:
                    last_x = float(mxy.group(1))
                    last_y = float(mxy.group(2))

                mz = pat_z.search(ln)
                if mz:
                    last_z_local = float(mz.group(1))

        if last_x is None or last_y is None:
            return (None, None, None)

        z = last_z_local if last_z_local is not None else fallback_z
        return (last_x, last_y, z)

    out           = []
    last_z        = 0.2
    pending_cool  = None    # tool to cool immediately after next T
    i             = 0

    # --- main rewrite ---
    while i < len(lines):
        ln = lines[i]

        # track current Z globally
        if ln.startswith(('G0', 'G1')):
            mz = pat_z.search(ln)
            if mz:
                last_z = float(mz.group(1))

        mT = pat_T.match(ln)
        if mT:
            tool = int(mT.group(2))

            # 1) patch coordinates using "last travel before first extrusion"
            x, y, z = find_target(i + 1, last_z)
            if x is not None:
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

    # --- dangling prime-tower prune (unchanged) ---
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
