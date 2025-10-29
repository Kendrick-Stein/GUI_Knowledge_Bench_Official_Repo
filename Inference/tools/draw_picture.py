# pylint: disable=no-member
from typing import Optional, Tuple, Dict, Any, List, Union
import os

import cv2
import numpy as np


Color = Tuple[int, int, int]  # BGR


def _normalize_bbox(value: Union[List[int], Tuple[int, int, int, int], None]) -> Optional[Tuple[int, int, int, int]]:
    """
    Normalize bbox to (x1, y1, x2, y2) and ensure x1 <= x2, y1 <= y2.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(int, value)
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _normalize_point(value: Union[List[int], Tuple[int, int], None]) -> Optional[Tuple[int, int]]:
    """
    Normalize point to (x, y).
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = map(int, value)
    except Exception:
        return None
    return (x, y)


def _clamp_point(pt: Tuple[int, int], w: int, h: int) -> Tuple[int, int]:
    x, y = pt
    return (max(0, min(w - 1, x)), max(0, min(h - 1, y)))


def _clamp_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _default_save_path(image_path: str) -> str:
    base, ext = os.path.splitext(image_path)
    if not ext:
        ext = ".png"
    return f"{base}_ann{ext}"


def draw_annotation(
    image_path: str,
    annotation: Dict[str, Any],
    save_path: Optional[str] = None,
    color: Color = (0, 0, 255),
    thickness: int = 1,
    point_radius: int = 5,
) -> str:
    """
    Draw annotation content onto the image and save.

    Parameters:
    - image_path: input image path
    - annotation: dict that may include:
        {
            "bbox": [x1, y1, x2, y2],  # optional
            "point": [x, y]            # optional
        }
      Both keys may co-exist.
    - save_path: output path; if omitted, appends "_ann" to the original filename
    - color: drawing color (B, G, R), default red
    - thickness: border/outline thickness
    - point_radius: visual radius for the point (outer ring); inner ring is half and filled

    Returns:
    - The actual saved file path
    """
    if not isinstance(annotation, dict):
        raise TypeError("annotation must be a dict")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    canvas = img.copy()

    # Parse inputs
    bbox_val = annotation.get("bbox", None)
    point_val = annotation.get("point", None)

    bbox = _normalize_bbox(bbox_val) if bbox_val is not None else None
    point = _normalize_point(point_val) if point_val is not None else None

    # Draw bbox
    if bbox is not None:
        x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
        # If side length is zero, slightly expand to ensure visibility
        if x1 == x2:
            x2 = min(w - 1, x2 + 1)
        if y1 == y2:
            y2 = min(h - 1, y2 + 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    # Draw point
    if point is not None:
        px, py = _clamp_point(point, w, h)
        # Outer ring
        cv2.circle(canvas, (px, py), max(1, int(point_radius)), color, max(1, int(thickness)), lineType=cv2.LINE_AA)
        # Inner filled circle
        cv2.circle(canvas, (px, py), max(1, int(point_radius // 2)), color, -1, lineType=cv2.LINE_AA)

    # Save even if neither bbox nor point was valid (keep pipeline stable)
    out_path = save_path or _default_save_path(image_path)
    ok = cv2.imwrite(out_path, canvas)
    if not ok:
        raise OSError(f"Failed to write output image: {out_path}")
    return out_path



from typing import Optional, Tuple, List, Union, Dict

import cv2
import numpy as np
import math
import re


Color = Tuple[int, int, int]  # BGR
BBox = Tuple[int, int, int, int]


def _get_center(bbox: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) // 2), int((y1 + y2) // 2))


def _clamp_point(pt: Tuple[int, int], w: int, h: int) -> Tuple[int, int]:
    x, y = pt
    return (max(0, min(w - 1, x)), max(0, min(h - 1, y)))


def _normalize_bbox(b: Union[List[int], Tuple[int, int, int, int]]) -> Optional[BBox]:
    """
    Ensure bbox is 4-int tuple (x1,y1,x2,y2). Auto-swap if x2 < x1 or y2 < y1.
    """
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(int, b)
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _normalize_point(p: Union[List[int], Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """
    Ensure point is 2-int tuple (x, y).
    """
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        return None
    try:
        x, y = map(int, p)
    except Exception:
        return None
    return (x, y)


def _color_for_option(option: Optional[str]) -> Color:
    """
    Return color for option letter. Default to RED if None/invalid.
    A: RED, B: BLUE, C: GREEN, D: ORANGE
    """
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 180, 0)
    ORANGE = (0, 165, 255)
    if not option:
        return RED
    o = str(option).strip().upper()
    if o == "A":
        return RED
    if o == "B":
        return BLUE
    if o == "C":
        return GREEN
    if o == "D":
        return ORANGE
    return RED


def _draw_text_with_bg(
    img: np.ndarray,
    text_lines: List[str],
    org: Tuple[int, int],
    text_color: Color = (255, 255, 255),   # white text for contrast
    bg_color: Color = (0, 0, 255),         # default red background (BGR)
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    thickness: int = 2,
    padding: int = 6,
    line_gap: int = 4,
) -> Tuple[int, int, int, int]:
    """
    Draw multiline text with a solid background rectangle for legibility.
    org: proposed top-left inside padding (i.e., where text will start).
    Returns drawn rectangle (x1, y1, x2, y2).
    """
    h_img, w_img = img.shape[:2]

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines]
    max_w = max((sz[0] for sz in sizes), default=0)
    line_h = max((sz[1] for sz in sizes), default=int(16 * font_scale))
    total_h = sum(line_h for _ in text_lines) + line_gap * (len(text_lines) - 1)

    rect_w = max_w + 2 * padding
    rect_h = total_h + 2 * padding

    x, y = org
    # Clamp rectangle in bounds
    x = max(0, min(w_img - rect_w, x))
    y = max(0, min(h_img - rect_h, y))

    # Background
    cv2.rectangle(img, (x, y), (x + rect_w, y + rect_h), bg_color, -1)

    # Draw lines
    cur_y = y + padding + line_h
    for line in text_lines:
        cv2.putText(img, line, (x + padding, cur_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cur_y += line_h + line_gap

    return (x, y, x + rect_w, y + rect_h)


# ---------- Label placement helpers (to avoid overlap with arrow and among options) ----------

def _measure_text_rect(
    text_lines: List[str],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    thickness: int = 2,
    padding: int = 6,
    line_gap: int = 4,
) -> Tuple[int, int]:
    """
    Compute the background rectangle size (width, height) for given text settings.
    """
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines]
    max_w = max((sz[0] for sz in sizes), default=0)
    line_h = max((sz[1] for sz in sizes), default=int(16 * font_scale))
    total_h = sum(line_h for _ in text_lines) + line_gap * (len(text_lines) - 1)
    rect_w = max_w + 2 * padding
    rect_h = total_h + 2 * padding
    return rect_w, rect_h


def _segment_point_distance(a: Tuple[int, int], b: Tuple[int, int], p: Tuple[float, float]) -> float:
    """
    Minimum distance from point p to line segment a-b.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 == 0:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab_len2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def _point_rect_distance(pt: Tuple[int, int], rect: Tuple[int, int, int, int]) -> float:
    """
    Minimum distance from point pt to axis-aligned rectangle rect=(x1,y1,x2,y2).
    If inside, returns 0.
    """
    px, py = pt
    x1, y1, x2, y2 = rect
    dx = 0
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2
    dy = 0
    if py < y1:
        dy = y1 - py
    elif py > y2:
        dy = py - y2
    if dx == 0 and dy == 0:
        return 0.0
    return math.hypot(dx, dy)


def _quadrant_order(preferred: str) -> List[str]:
    order_all = ["NE", "SE", "SW", "NW"]
    preferred = preferred if preferred in order_all else "NE"
    i = order_all.index(preferred)
    return order_all[i:] + order_all[:i]


def _best_label_org(
    end_pt: Tuple[int, int],
    start_pt: Tuple[int, int],
    img_shape: Tuple[int, int, int],
    text_lines: List[str],
    preferred_option: Optional[str],
    base_offset: float = 16.0,
    min_dist: float = 22.0,
    max_expand_iters: int = 3,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1,
    thickness: int = 3,
    padding: int = 6,
    line_gap: int = 4,
    protect_end_radius: float = 0.0,
) -> Tuple[int, int]:
    """
    Choose a top-left origin for the label near end_pt, avoiding overlap with the arrow and staying on-screen.
    - preferred_option maps to quadrant: A->NE, B->SE, C->SW, D->NW
    - protect_end_radius: enforce a minimum distance between the label's background and the end_pt,
      used to prevent the label from covering the center and highlight ring for point-based actions.
    """
    h, w = img_shape[:2]
    rect_w, rect_h = _measure_text_rect(text_lines, font, font_scale, thickness, padding, line_gap)

    opt = (preferred_option or "").strip().upper()
    quadrants = {"A": "NE", "B": "SE", "C": "SW", "D": "NW"}
    preferred_quad = quadrants.get(opt, "NE")
    quad_order = _quadrant_order(preferred_quad)

    # Arrow segment for distance check
    a = (int(start_pt[0]), int(start_pt[1]))
    b = (int(end_pt[0]), int(end_pt[1]))

    # Unit vectors for quadrants
    quad_vec = {
        "NE": (1.0, -1.0),
        "SE": (1.0, 1.0),
        "SW": (-1.0, 1.0),
        "NW": (-1.0, -1.0),
    }

    scale = 1.0
    best_org = (max(0, min(w - rect_w, end_pt[0] + 10)), max(0, min(h - rect_h, end_pt[1] - 10)))
    for _ in range(max_expand_iters):
        offset = base_offset * scale
        for quad in quad_order:
            sx, sy = quad_vec[quad]
            # Candidate center position near end_pt
            cx = end_pt[0] + int(round(sx * offset))
            cy = end_pt[1] + int(round(sy * offset))
            # Convert to top-left origin and clamp to screen
            x = max(0, min(w - rect_w, cx - rect_w // 2))
            y = max(0, min(h - rect_h, cy - rect_h // 2))
            # Recompute rect center after clamping
            rcx = x + rect_w / 2.0
            rcy = y + rect_h / 2.0
            # Keep away from the arrow line
            dist_line = _segment_point_distance(a, b, (rcx, rcy))
            # And keep the label background away from end point by protect_end_radius
            dist_pt_rect = _point_rect_distance(end_pt, (x, y, x + rect_w, y + rect_h))
            if dist_line >= min_dist and dist_pt_rect >= protect_end_radius:
                return (int(x), int(y))
            # Otherwise try next quadrant
        scale *= 1.5  # expand outward and retry
        best_org = (max(0, min(w - rect_w, end_pt[0] + int(10 * scale))),
                    max(0, min(h - rect_h, end_pt[1] - int(10 * scale))))
    return (int(best_org[0]), int(best_org[1]))


# ------------------------- DSL Parsing Helpers -------------------------

_action_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(.*?)\s*\)\s*$")


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1]
    # Unescape common sequences
    s = s.replace(r"\\", "\0")  # temporary
    s = s.replace(r"\'", "'").replace(r"\"", '"').replace(r"\n", "\n").replace(r"\t", "\t").replace(r"\r", "\r")
    s = s.replace("\0", "\\")
    return s


def _split_args(arg_str: str) -> List[str]:
    """
    Split "k1='v1', k2=\"v2 with ,\", k3='v3'" into ["k1='v1'", "k2=\"v2 with ,\"", "k3='v3'"].
    Handles quotes and escapes.
    """
    parts = []
    cur = []
    in_quote = None
    esc = False
    for ch in arg_str:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if ch == "\\":
            cur.append(ch)
            esc = True
            continue
        if in_quote:
            cur.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            cur.append(ch)
            in_quote = ch
            continue
        if ch == ",":
            part = "".join(cur).strip()
            if part:
                parts.append(part)
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_kv_pairs(arg_str: str) -> Dict[str, str]:
    """
    Parse comma-separated key=value pairs into dict of raw string values (still quoted).
    """
    if not arg_str.strip():
        return {}
    result: Dict[str, str] = {}
    for piece in _split_args(arg_str):
        if "=" not in piece:
            raise ValueError(f"Invalid argument segment: {piece}")
        k, v = piece.split("=", 1)
        key = k.strip()
        val = v.strip()
        if not key:
            raise ValueError(f"Empty key in segment: {piece}")
        result[key] = val
    return result


def _parse_point_str(s: str) -> Tuple[int, int]:
    """
    Convert 'x y' string to (x, y) ints.
    """
    s = _strip_quotes(s)
    toks = s.strip().split()
    if len(toks) != 2:
        raise ValueError(f"Point must be 'x y', got: {s!r}")
    x, y = toks
    return (int(x), int(y))


def _parse_action_desc(desc: str) -> Tuple[str, Dict[str, Union[str, Tuple[int, int]]]]:
    """
    Return (action_name_lower, args_dict with parsed values where appropriate).
    """
    m = _action_re.match(desc or "")
    if not m:
        raise ValueError(f"Invalid action description: {desc!r}")
    name = m.group(1).strip().lower()
    arg_str = m.group(2)

    raw = _parse_kv_pairs(arg_str)

    args: Dict[str, Union[str, Tuple[int, int]]] = {}

    # Helpers
    def has(k: str) -> bool:
        return k in raw

    # Parse according to possible keys
    if has("point"):
        args["point"] = _parse_point_str(raw["point"])
    if has("start_point"):
        args["start_point"] = _parse_point_str(raw["start_point"])
    if has("end_point"):
        args["end_point"] = _parse_point_str(raw["end_point"])
    if has("direction"):
        args["direction"] = _strip_quotes(raw["direction"]).strip().lower()
    if has("content"):
        args["content"] = _strip_quotes(raw["content"])
    if has("key"):
        args["key"] = _strip_quotes(raw["key"])

    return name, args


# ------------------------- Drawing via DSL -------------------------

def draw_action(img: np.ndarray, action_desc: str, option: Optional[str] = None) -> np.ndarray:
    """
    Draw a GUI action visualization on the image using a DSL string.

    action_desc examples (strings):
      - "click(point='120 220')"
      - "left_double(point='520 120')"
      - "right_single(point='100 380')"
      - "long_press(point='540 380')"
      - "drag(start_point='100 100', end_point='200 100')"
      - "scroll(point='540 100', direction='down')"
      - "type(content='Hello\\n')"  # no drawing
      - "hotkey(key='ctrl c')"      # no drawing

    option:
      - One of {'A','B','C','D'} (case-insensitive). If provided, the label will be prefixed as "(A) Click" etc.
    """
    canvas = img.copy()
    h, w = canvas.shape[:2]

    # Unified scaling
    short_side = min(h, w)
    scale = short_side / 800.0
    scale = min(max(scale, 0.5), 2.0)

    def s(v: float) -> int:
        # scale for pixel-based values (radius, thickness, padding, gaps, offsets)
        return max(1, int(round(v * scale)))

    def sf(v: float) -> float:
        # scale for font size with clamps to keep legible and not oversized
        return max(0.5, min(1.6, v * scale))

    # Common text style
    font_scale = sf(0.6)
    thick_text = s(2)
    padding = s(6)
    line_gap = max(2, s(4))

    def _with_option_prefix(text: str) -> str:
        if option is None:
            return text
        opt = str(option).strip().upper()
        if opt in {"A", "B", "C", "D"}:
            return f"({opt}) {text}"
        return text

    COLOR: Color = _color_for_option(option)

    name, args = _parse_action_desc(action_desc)
    t = (name or "").strip().lower()

    # --------- Non-drawing actions (no-ops visually) ---------
    if t in ("type", "hotkey"):
        return canvas

    # --------- Point-based clicks ---------
    if t in ("click", "left_double", "right_single", "long_press"):
        if "point" not in args:
            raise ValueError(f"{t} requires point='x y'")
        center = _clamp_point(args["point"], w, h)  # type: ignore[arg-type]

        # unified indicator (small dot + thin ring) and ABCD-only label
        ring_radius = s(9)
        ring_thickness = max(1, s(1))
        dot_radius = s(2)
        cv2.circle(canvas, center, ring_radius, COLOR, ring_thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, dot_radius, COLOR, -1)

        lbl = (str(option).strip().upper() if option and str(option).strip().upper() in {"A","B","C","D"} else "")
        if lbl:
            font_scale_lbl = max(0.35, font_scale * 0.70)
            label_org = _best_label_org(
                end_pt=center,
                start_pt=center,
                img_shape=canvas.shape,
                text_lines=[lbl],
                preferred_option=lbl,
                base_offset=s(12),
                min_dist=s(18),
                font_scale=font_scale_lbl,
                thickness=thick_text,
                padding=max(1, padding // 3),
                line_gap=line_gap,
                protect_end_radius=float(ring_radius + s(6)),
            )
            _draw_text_with_bg(
                canvas,
                [lbl],
                label_org,
                text_color=(255, 255, 255),
                bg_color=COLOR,
                font_scale=font_scale_lbl,
                thickness=thick_text,
                padding=max(1, padding // 3),
                line_gap=line_gap,
            )
        return canvas


    # --------- Drag (arrow from start to end) ---------
    if t == "drag":
        if "start_point" not in args or "end_point" not in args:
            raise ValueError("drag requires start_point='x y' and end_point='x y'")
        start_pt = _clamp_point(args["start_point"], w, h)  # type: ignore[arg-type]
        end_pt = _clamp_point(args["end_point"], w, h)      # type: ignore[arg-type]
        cv2.arrowedLine(canvas, start_pt, end_pt, COLOR, s(3), tipLength=0.2)

        # Only show small A/B/C/D label when option provided
        lbl = (str(option).strip().upper() if option and str(option).strip().upper() in {"A","B","C","D"} else "")
        if lbl:
            font_scale_lbl = max(0.35, font_scale * 0.70)
            label_org = _best_label_org(
                end_pt=end_pt,
                start_pt=start_pt,
                img_shape=canvas.shape,
                text_lines=[lbl],
                preferred_option=lbl,
                base_offset=s(12),
                min_dist=s(18),
                font_scale=font_scale_lbl,
                thickness=thick_text,
                padding=max(1, padding // 3),
                line_gap=line_gap,
            )
            _draw_text_with_bg(
                canvas,
                [lbl],
                label_org,
                text_color=(255, 255, 255),
                bg_color=COLOR,
                font_scale=font_scale_lbl,
                thickness=thick_text,
                padding=max(1, padding // 3),
                line_gap=line_gap,
            )
        return canvas

    # --------- Scroll (arrow in given direction from point) ---------
    if t == "scroll":
        if "point" not in args or "direction" not in args:
            raise ValueError("scroll requires point='x y' and direction in {'up','down','left','right'}")
        direction = str(args["direction"]).lower()
        base_scroll = max(15, s(50))  # ensure minimum displacement on tiny images

        dir_map = {
            "up": (0, -base_scroll),
            "down": (0, base_scroll),
            "left": (-base_scroll, 0),
            "right": (base_scroll, 0),
        }
        if direction not in dir_map:
            raise ValueError("scroll direction must be one of {'up','down','left','right'}")
        dx, dy = dir_map[direction]

        center = _clamp_point(args["point"], w, h)  # type: ignore[arg-type]
        end = _clamp_point((center[0] + dx, center[1] + dy), w, h)
        cv2.arrowedLine(canvas, center, end, COLOR, s(2), tipLength=0.3)

        # Only show small A/B/C/D label when option provided
        lbl = (str(option).strip().upper() if option and str(option).strip().upper() in {"A","B","C","D"} else "")
        if lbl:
            font_scale_lbl = max(0.35, font_scale * 0.70)
            label_org = _best_label_org(
                end_pt=end,
                start_pt=center,
                img_shape=canvas.shape,
                text_lines=[lbl],
                preferred_option=lbl,
                base_offset=s(12),
                min_dist=s(18),
                font_scale=font_scale_lbl,
                thickness=thick_text,
                padding=max(1, padding // 3),
                line_gap=line_gap,
            )
            _draw_text_with_bg(
                canvas,
                [lbl],
                label_org,
                text_color=(255, 255, 255),
                bg_color=COLOR,
                font_scale=font_scale_lbl,
                thickness=thick_text,
                padding=max(1, padding // 3),
                line_gap=line_gap,
            )
        return canvas

    # Unknown action: leave image as-is
    return canvas


def _ensure_image_exists(path: str, fallback_size: Tuple[int, int] = (480, 640)) -> np.ndarray:
    img = cv2.imread(path)
    if img is not None:
        return img
    # Generate a plain white image as fallback
    h, w = fallback_size
    return np.full((h, w, 3), 255, dtype=np.uint8)


if __name__ == "__main__":
    # Demo script: applies supported action types and writes outputs using DSL
    base_img = _ensure_image_exists("test.png")

    # Individual demos
    demos = [
        ("click(point='100 200')", None, "out_0_click.png"),
        ("left_double(point='200 320')", None, "out_1_left_double.png"),
        ("right_single(point='300 420')", None, "out_2_right_single.png"),
        ("drag(start_point='100 200', end_point='450 480')", None, "out_3_drag.png"),
        ("scroll(point='300 320', direction='down')", None, "out_4_scroll.png"),
        ("long_press(point='400 260')", None, "out_5_long_press.png"),
    ]

    for action_desc, opt, outfile in demos:
        out = draw_action(base_img, action_desc, opt)
        cv2.imwrite(outfile, out)

    # Combined ABCD test on the same image (clicks)
    img_abcd = base_img.copy()
    img_abcd = draw_action(img_abcd, "click(point='100 100')", "A")          # top-left -> "(A) Click" (RED)
    img_abcd = draw_action(img_abcd, "click(point='540 100')", "B")          # top-right -> "(B) Click" (BLUE)
    img_abcd = draw_action(img_abcd, "click(point='100 380')", "C")          # bottom-left -> "(C) Click" (GREEN)
    img_abcd = draw_action(img_abcd, "click(point='540 380')", "D")          # bottom-right -> "(D) Click" (ORANGE)
    cv2.imwrite("out_abcd.png", img_abcd)

    # Combined ABCD test for left_double
    img_left_double = base_img.copy()
    img_left_double = draw_action(img_left_double, "left_double(point='100 100')", "A")
    img_left_double = draw_action(img_left_double, "left_double(point='540 100')", "B")
    img_left_double = draw_action(img_left_double, "left_double(point='100 380')", "C")
    img_left_double = draw_action(img_left_double, "left_double(point='540 380')", "D")
    cv2.imwrite("out_left_double_abcd.png", img_left_double)

    # Combined ABCD test for right_single
    img_right_single = base_img.copy()
    img_right_single = draw_action(img_right_single, "right_single(point='100 100')", "A")
    img_right_single = draw_action(img_right_single, "right_single(point='540 100')", "B")
    img_right_single = draw_action(img_right_single, "right_single(point='100 380')", "C")
    img_right_single = draw_action(img_right_single, "right_single(point='540 380')", "D")
    cv2.imwrite("out_right_single_abcd.png", img_right_single)

    # Combined ABCD test for long_press
    img_long_press = base_img.copy()
    img_long_press = draw_action(img_long_press, "long_press(point='100 100')", "A")
    img_long_press = draw_action(img_long_press, "long_press(point='540 100')", "B")
    img_long_press = draw_action(img_long_press, "long_press(point='100 380')", "C")
    img_long_press = draw_action(img_long_press, "long_press(point='540 380')", "D")
    cv2.imwrite("out_long_press_abcd.png", img_long_press)

    # Combined ABCD test for drag (four directions)
    img_drag = base_img.copy()
    img_drag = draw_action(img_drag, "drag(start_point='100 100', end_point='200 100')", "A")     # right (RED)
    img_drag = draw_action(img_drag, "drag(start_point='540 100', end_point='540 200')", "B")     # down (BLUE)
    img_drag = draw_action(img_drag, "drag(start_point='100 380', end_point='40 380')", "C")      # left (GREEN)
    img_drag = draw_action(img_drag, "drag(start_point='540 380', end_point='540 320')", "D")     # up (ORANGE)
    cv2.imwrite("out_drag_abcd.png", img_drag)

    # Combined ABCD test for scroll (points + directions)
    img_scroll = base_img.copy()
    img_scroll = draw_action(img_scroll, "scroll(point='100 100', direction='up')", "A")       # RED
    img_scroll = draw_action(img_scroll, "scroll(point='540 100', direction='down')", "B")     # BLUE
    img_scroll = draw_action(img_scroll, "scroll(point='100 380', direction='left')", "C")     # GREEN
    img_scroll = draw_action(img_scroll, "scroll(point='540 380', direction='right')", "D")    # ORANGE
    cv2.imwrite("out_scroll_abcd.png", img_scroll)

    # Drag overlap test (same direction, different lengths) to verify avoidance and color distinction
    img_drag_overlap = base_img.copy()
    img_drag_overlap = draw_action(img_drag_overlap, "drag(start_point='200 240', end_point='550 240')", "A")  # long right (RED)
    img_drag_overlap = draw_action(img_drag_overlap, "drag(start_point='220 240', end_point='360 240')", "B")  # short right (BLUE)
    cv2.imwrite("out_drag_overlap.png", img_drag_overlap)
