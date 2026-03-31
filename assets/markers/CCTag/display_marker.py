#!/usr/bin/env python3
"""
Full-screen CCTag marker display with controllable blink frequency
and 10-bit data encoded as LEFT/RIGHT asymmetric half-arc rings around the CCTag.

Structure (inside-out):
    CCTag core rings (fixed, for detection & ID)
    → Inner guard ring   (fixed black half-arcs, width = CCTag outermost ring width)
    → 5 data rings LEFT  (bit 9/MSB innermost … bit 5/LSB outermost, left semicircle)
    → 5 data rings RIGHT (bit 4/MSB innermost … bit 0/LSB outermost, right semicircle)
    → Outer guard ring   (fixed black half-arcs, width = guard width)

Left half-arcs  (90° → 270° sweeping through 180°, i.e. the left hemisphere)
Right half-arcs (270° → 90° sweeping through 0°,   i.e. the right hemisphere)

The left side encodes bits 9–5 (high nibble+1) and the right side encodes
bits 4–0 (low nibble+1).  Together they form a 10-bit value (0–1023).

Ring widths are fixed at 1.5× guard width for consistent decoding.
Outer data rings may extend beyond screen edges — graceful degradation
(LSBs are lost first, preserving the most significant bits).

Usage:
    python display_marker.py [--marker ID] [--rings {3,4}] [--freq HZ] [--duty RATIO] [--data VALUE]

Press 'q' or ESC to quit.
Press LEFT/RIGHT arrow keys to switch markers.
Press UP/DOWN arrow keys to increase/decrease blink frequency.
Press PageUp/PageDown or A/D keys to increase/decrease 10-bit data value (0–1023).
"""

import argparse
import os
import sys
import tkinter as tk


# --------------------------------------------------------------------------- #
# Marker data – loaded from generate.py's source txt files                    #
# --------------------------------------------------------------------------- #

def _load_marker_txt(filename):
    """Load marker radii from cctag3.txt / cctag4.txt (same format as generate.py)."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "markersToPrint", "generators", filename)
    markers = []
    try:
        with open(base) as f:
            for line in f:
                vals = [int(x) for x in line.split()]
                if vals:
                    markers.append(vals)
    except FileNotFoundError:
        pass
    return markers


CCTAG3 = _load_marker_txt("cctag3.txt")
CCTAG4 = _load_marker_txt("cctag4.txt")


BITS_PER_SIDE = 5   # 5 left + 5 right = 10 bits total
DATA_MAX      = (1 << (BITS_PER_SIDE * 2)) - 1  # 1023


# --------------------------------------------------------------------------- #
# Drawing helpers                                                              #
# --------------------------------------------------------------------------- #

def draw_marker(canvas, rings, cx, cy, radius, blacked=False):
    """Draw a CCTag marker on *canvas* centred at (cx, cy) with outer *radius*.

    Same colour scheme as generate.py (black outer disc, rings alternate
    white/black starting with white). Background is white.

    *blacked* – when True, all white parts become black (entire marker = black).
    """
    canvas.delete("marker")

    scale = radius / 100.0

    # Outermost black disc
    canvas.create_oval(cx - radius, cy - radius,
                       cx + radius, cy + radius,
                       fill="black", outline="", tags="marker")

    # Inner rings: white → black → white → black → white (innermost = white)
    # When blacked=True, white becomes black (everything is black)
    fill_color = "white"
    for r in rings:
        r_px = scale * r
        actual_color = "black" if (blacked and fill_color == "white") else fill_color
        canvas.create_oval(cx - r_px, cy - r_px,
                           cx + r_px, cy + r_px,
                           fill=actual_color, outline="", tags="marker")
        fill_color = "black" if fill_color == "white" else "white"


def draw_data_rings(canvas, data_val, cx, cy, radius,
                    guard_width, ring_width, sep_width=2):
    """Draw 10-bit data as LEFT/RIGHT asymmetric half-arc rings outside the CCTag.

    Structure (inside-out on each side):
        CCTag outer disc (radius)
        → inner guard half-arc  (fixed black, width = guard_width)
        → 5 data half-arcs      (dynamic black/white, width = ring_width each,
                                  with gray separator arcs between adjacent rings)
        → outer guard half-arc  (fixed black, width = guard_width)

    LEFT  side (arc sweeping through 180°, left hemisphere):
        Encodes HIGH 5 bits — bit 9 (MSB) innermost … bit 5 (LSB) outermost.

    RIGHT side (arc sweeping through 0°, right hemisphere):
        Encodes LOW 5 bits — bit 4 (MSB) innermost … bit 0 (LSB) outermost.

    Black arc = bit 1, white arc = bit 0.
    Gray separator arcs (sep_width pixels) between each pair of data arcs.

    These arcs are NOT affected by blink – colours always reflect bit values.
    """
    canvas.delete("dataring")

    n = BITS_PER_SIDE  # 5
    high_bits = (data_val >> n) & ((1 << n) - 1)  # bits 9–5
    low_bits  =  data_val       & ((1 << n) - 1)  # bits 4–0

    bits_left  = [(high_bits >> i) & 1 for i in range(n)]  # [0]=b5 .. [4]=b9
    bits_right = [(low_bits  >> i) & 1 for i in range(n)]  # [0]=b0 .. [4]=b4

    # Left  half: tkinter arc start=90,  extent=180  (sweeps 90→270 through 180°)
    # Right half: tkinter arc start=270, extent=180  (sweeps 270→90 through 0°)
    sides = [
        ("left",  bits_left,  90,  180),
        ("right", bits_right, 270, 180),
    ]

    data_zone_start = radius + guard_width  # inner edge of first data ring

    for side_name, bits, arc_start, arc_extent in sides:
        # 1. Outer guard arc (black) – drawn first (largest)
        last_ring_end = data_zone_start + n * ring_width + (n - 1) * sep_width
        r_outer_guard = last_ring_end + guard_width
        canvas.create_arc(
            cx - r_outer_guard, cy - r_outer_guard,
            cx + r_outer_guard, cy + r_outer_guard,
            start=arc_start, extent=arc_extent,
            fill="black", outline="", style=tk.CHORD,
            tags="dataring"
        )

        # 2. Data arcs + separators, from outermost (LSB) to innermost (MSB)
        for k in range(n - 1, -1, -1):
            bit_idx = (n - 1) - k    # k=(n-1)→bit[0] (LSB/outermost), k=0→bit[n-1] (MSB)
            ring_start = data_zone_start + k * (ring_width + sep_width)
            ring_end   = ring_start + ring_width

            # Gray separator arc just outside this data arc (except after outermost)
            if k < n - 1:
                r_sep = ring_end + sep_width
                canvas.create_arc(
                    cx - r_sep, cy - r_sep,
                    cx + r_sep, cy + r_sep,
                    start=arc_start, extent=arc_extent,
                    fill="#888888", outline="", style=tk.CHORD,
                    tags="dataring"
                )

            # Data arc
            fill = "black" if bits[bit_idx] else "white"
            canvas.create_arc(
                cx - ring_end, cy - ring_end,
                cx + ring_end, cy + ring_end,
                start=arc_start, extent=arc_extent,
                fill=fill, outline="", style=tk.CHORD,
                tags="dataring"
            )

        # 3. Inner guard arc (black) – drawn last (smallest in data-arc group)
        r_inner_guard = data_zone_start
        canvas.create_arc(
            cx - r_inner_guard, cy - r_inner_guard,
            cx + r_inner_guard, cy + r_inner_guard,
            start=arc_start, extent=arc_extent,
            fill="black", outline="", style=tk.CHORD,
            tags="dataring"
        )


# --------------------------------------------------------------------------- #
# Main application                                                             #
# --------------------------------------------------------------------------- #

class MarkerApp:
    def __init__(self, root, args):
        self.root = root
        self.freq = args.freq        # blink frequency in Hz (0 = always on)
        self.duty = args.duty        # duty cycle 0-1 (fraction of period ON)
        self.rings_count = args.rings
        self.marker_id = args.marker
        self.data = args.data        # 10-bit data value (0–1023)
        self.inverted = False        # False = normal, True = inverted (blink phase)
        self._blink_job = None

        markers = CCTAG3 if self.rings_count == 3 else CCTAG4
        if not markers:
            print("ERROR: 4-ring marker data not available.", file=sys.stderr)
            sys.exit(1)
        self.markers = markers
        self.marker_id = max(0, min(self.marker_id, len(markers) - 1))

        # ---- Window setup ------------------------------------------------- #
        root.title("CCTag Marker Display")
        root.configure(bg="white")
        root.overrideredirect(True)
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{sw}x{sh}+0+0")

        self.canvas = tk.Canvas(root, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ---- Key bindings ------------------------------------------------- #
        root.bind("<q>",      lambda e: self._quit())
        root.bind("<Escape>", lambda e: self._quit())
        root.bind("<Left>",   lambda e: self._prev_marker())
        root.bind("<Right>",  lambda e: self._next_marker())
        root.bind("<Up>",     lambda e: self._increase_freq())
        root.bind("<Down>",   lambda e: self._decrease_freq())
        root.bind("<Prior>",  lambda e: self._increase_data())   # PageUp
        root.bind("<Next>",   lambda e: self._decrease_data())   # PageDown
        root.bind("<a>",      lambda e: self._decrease_data())
        root.bind("<d>",      lambda e: self._increase_data())

        # ---- Initial draw ------------------------------------------------- #
        root.update_idletasks()
        self._resize_and_draw()
        self.canvas.bind("<Configure>", lambda e: self._resize_and_draw())

        self._schedule_blink()

    # ----------------------------------------------------------------------- #
    # Geometry                                                                 #
    # ----------------------------------------------------------------------- #

    def _resize_and_draw(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.cx = w // 2
        self.cy = h // 2
        self.radius = int(min(w, h) * 0.45)

        # Guard ring width = width of the CCTag's outermost ring.
        rings = self.markers[self.marker_id]
        scale = self.radius / 100.0
        self.guard_width = max(2, int(self.radius - scale * rings[0]))

        # Data rings are 1.5× the guard width for better far-distance readability.
        # Outer rings may extend beyond screen edges — graceful degradation
        # (LSBs are lost first, preserving the most significant bits).
        self.ring_width = max(3, int(self.guard_width * 1.5))

        self._redraw()

    def _redraw(self):
        if self.inverted:
            # Blink phase: everything black (background + marker)
            self.canvas.configure(bg="black")
            self.root.configure(bg="black")
        else:
            # Normal phase: white background + normal marker
            self.canvas.configure(bg="white")
            self.root.configure(bg="white")
        # Draw data rings first (outside CCTag), then CCTag on top so its
        # outer disc cleanly covers the guard ring's inner edge.
        draw_data_rings(self.canvas, self.data,
                        self.cx, self.cy, self.radius,
                        self.guard_width, self.ring_width)
        draw_marker(self.canvas,
                    self.markers[self.marker_id],
                    self.cx, self.cy, self.radius,
                    blacked=self.inverted)


    # ----------------------------------------------------------------------- #
    # Blinking logic                                                           #
    # ----------------------------------------------------------------------- #

    def _schedule_blink(self):
        if self._blink_job is not None:
            self.root.after_cancel(self._blink_job)
            self._blink_job = None

        if self.freq <= 0:
            # Always on – no blinking
            self.inverted = False
            self._redraw()
            return

        period_ms = 1000.0 / self.freq
        if not self.inverted:
            # Normal phase → stay for duty * period, then invert
            delay = int(period_ms * self.duty)
        else:
            # Inverted phase → stay for (1-duty) * period, then restore
            delay = int(period_ms * (1.0 - self.duty))

        delay = max(delay, 1)
        self._blink_job = self.root.after(delay, self._toggle)

    def _toggle(self):
        self.inverted = not self.inverted
        self._redraw()
        self._schedule_blink()

    # ----------------------------------------------------------------------- #
    # Controls                                                                 #
    # ----------------------------------------------------------------------- #

    def _prev_marker(self):
        self.marker_id = (self.marker_id - 1) % len(self.markers)
        self._redraw()

    def _next_marker(self):
        self.marker_id = (self.marker_id + 1) % len(self.markers)
        self._redraw()

    def _increase_freq(self):
        if self.freq <= 0:
            self.freq = 0.5
        elif self.freq < 1:
            self.freq = round(self.freq + 0.5, 1)
        elif self.freq < 10:
            self.freq = round(self.freq + 1.0, 1)
        else:
            self.freq = round(self.freq + 5.0, 1)
        self.inverted = False
        self._schedule_blink()

    def _decrease_freq(self):
        if self.freq <= 0.5:
            self.freq = 0.0
            self.inverted = False
            self._schedule_blink()
            return
        elif self.freq <= 1:
            self.freq = round(self.freq - 0.5, 1)
        elif self.freq <= 10:
            self.freq = round(self.freq - 1.0, 1)
        else:
            self.freq = round(self.freq - 5.0, 1)
        self.freq = max(0.0, self.freq)
        self.inverted = False
        self._schedule_blink()

    def _increase_data(self):
        self.data = min(DATA_MAX, self.data + 1)
        self._redraw()

    def _decrease_data(self):
        self.data = max(0, self.data - 1)
        self._redraw()

    def _quit(self):
        self.root.destroy()


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-screen CCTag marker display with controllable blink frequency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--marker", "-m", metavar="ID", type=int, default=0,
        help="marker ID to display (0-based index)",
    )
    parser.add_argument(
        "--rings", "-r", metavar="N", type=int, default=3, choices=[3, 4],
        help="number of rings {3, 4}",
    )
    parser.add_argument(
        "--freq", "-f", metavar="HZ", type=float, default=0.0,
        help="blink frequency in Hz; 0 = always on",
    )
    parser.add_argument(
        "--duty", "-d", metavar="RATIO", type=float, default=0.5,
        help="duty cycle (0.0–1.0): fraction of period the marker is visible",
    )
    parser.add_argument(
        "--data", metavar="VALUE", type=int, default=0,
        help="10-bit data value (0–1023) shown as left/right half-arc data rings",
    )
    args = parser.parse_args()

    if not 0.0 <= args.duty <= 1.0:
        parser.error("--duty must be between 0.0 and 1.0")
    if args.freq < 0:
        parser.error("--freq must be >= 0")
    if not 0 <= args.data <= DATA_MAX:
        parser.error(f"--data must be between 0 and {DATA_MAX}")

    return args


if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = MarkerApp(root, args)
    root.mainloop()
