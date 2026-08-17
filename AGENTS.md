# AGENTS.md

## Current state

- `CompVi.py`: real-time webcam hand-tracking with a Sci-Fi HUD / cyberpunk UI (MediaPipe HandLandmarker + OpenCV). All drawing is custom (`mp_draw` is NOT used): neon skeleton glove, pulsing fingertip energy nodes, multi-color laser cables between the two hands (per-finger colors from `FINGER_LASER`), `cv2.ellipse` angle arcs at thumb/index joints with degree text, faint "data scanning" coordinates, and a bloom layer (`draw_glow_line`/`draw_glow_circle` + `GaussianBlur`) for true neon glow. No HUD/text beyond the sci-fi elements.
- Contact trigger: when left↔right index OR thumb fingertips are < `CONTACT_DIST` (40 px), `apply_contact_blur` blurs only the ROI between the hands (GaussianBlur + mask restore so hands stay sharp — do NOT blur the full frame), `draw_radial_burst` flashes at the contact point, and laser cables pulse thicker/brighter (`pulse` passed into `draw_inter_hand_mesh`).
- Virtual Glass Frame replaced by **Multi-Zone Prism Lens** (2 hands only): 10 fingertips (5 L + 5 R) form 4 connected quad zones from adjacent finger pairs (`ZONE_SPECS` in BGR: thumb-index `COLORMAP_JET` thermal, index-middle `COLORMAP_OCEAN` cyber blue, middle-ring `COLORMAP_SUMMER` matrix green, ring-pinky `COLORMAP_MAGMA` synthwave) via `build_zone_quads`/`build_zone_quad` -> `order_quad_corners`. `render_zone_lens` perspective-crops each zone's ROI (`cv2.getPerspectiveTransform`/`warpPerspective` to `GLASS_WARP_SIZE`, inverse-warp + `fillPoly` mask + `cv2.dilate`; skips safely on out-of-bounds/degenerate quads) and applies the zone colormap + shared glass shimmer (`_make_glass_overlay` cyan→purple gradient, cached in `_GLASS_OVERLAY_CACHE`) via `cv2.addWeighted`. `render_prism_lens` runs all 4 zones per frame. `draw_prism_frame` draws dim cyan vertical zone sides (`draw_zone_edges`) + colored L corner brackets at all 10 fingertips (`draw_zone_brackets` -> `draw_corner_bracket`, color from `FINGER_LASER`: thumb yellow, index red, middle cyan, ring green, pinky magenta). The laser mesh `draw_inter_hand_mesh` draws the colored horizontal bars between each finger pair. Keys: `e` toggles prism lens, `f` toggles frame/brackets. All pure functions import-testable (`main()` only under `if __name__ == "__main__"`).
- `portal_filter.py`: `PortalFilterApp` class — hand-tracking portal filter app using `mp.tasks.vision.HandLandmarker` (Tasks API) + OpenCV. Gesture-driven: portal frame (4-corner polygon mask from thumb+index L/R fingertips applies active filter inside), slingshot (peace-left + pinch-right release fires filter to entire background), open-palm reset, thumb-pinky pinch cycles 5 filters (CYBER_NEON, SEPIA, NEGATIVE, THERMAL, PIXELATE). Keyboard: 1-5 select, SPACE cycle, r reset, h help, q/ESC quit. Camera selection via terminal at startup.
- `hand_landmarker.task`: MediaPipe model file (7.8 MB) required by the script; it auto-downloads if missing.
- No package manifest, dependency lockfile, test suite, linter config, or CI exists. Do not assume one.
- IS a git repository (has `.git/`).

## Python environment (important)

- The `python` on PATH is an MSYS2/ucrt64 build (Python 3.12). It is PEP-668 externally managed and pip on it cannot resolve Windows wheels (platform tag `mingw`), so it tries to build from source and fails. Do NOT use it to install or run this project.
- Use the python.org Python 3.14 instead. Its launcher works:
  - Run script: `py -3.14 CompVi.py`
  - Install packages: `py -3.14 -m pip install <pkg>`
- Dependencies already installed into Python 3.14: `opencv-python` (cv2 5.x), `mediapipe` (1.0.0), `numpy`.

## Working here

- `CompVi.py` runs via `py -3.14 CompVi.py`; press `q` in the window to exit. Runs `main()` under `if __name__ == "__main__"`, so it can be imported for testing pure functions without the webcam.
- `portal_filter.py` runs via `py -3.14 portal_filter.py`. Prompts for camera index selection at startup. Uses Tasks API (`mp.tasks.vision.HandLandmarker`) like CompVi.py. Pure methods (`apply_cyber_neon`, `apply_sepia`, etc.) are import-testable via `PortalFilterApp`.
- MediaPipe 1.x removed the legacy `mp.solutions` API. Both scripts use the Tasks API (`mp.tasks.vision.HandLandmarker`).
- `detect_for_video()` (CompVi) requires a monotonically increasing timestamp in ms and expects an `mp.Image` with `mp.ImageFormat.SRGB`.
- Webcam frame is 480x640 on this machine; landmarks are normalized `[0,1]` — multiply by frame w/h for pixels.
- Webcam quirk: the default MSMF backend opens but cannot grab frames (error `-1072875772`), so both scripts open with `cv2.VideoCapture(0, cv2.CAP_DSHOW)`.
- No tests/lint config; verify changes with `py -3.14 -m py_compile <script>` and a quick camera smoke test.
