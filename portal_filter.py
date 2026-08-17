"""
============================================================================
 Portal Filter - Hand Tracking + Portal FX (OpenCV / MediaPipe)
============================================================================
 Class  : PortalFilterApp
 Library: OpenCV, MediaPipe (Tasks API), NumPy
 Python : 3.14 (python.org)

 Fitur:
   1. Portal Frame  - Polygon mask dari 4 ujung jari (jempol + telunjuk
                       kiri & kanan); filter aktif diterapkan HANYA di
                       dalam area portal.
   2. Slingshot BG   - Tangan kiri pose Peace, tangan kanan Pinch
                       (jempol+telunjuk) ditarik. Saat pinch dilepas,
                       filter aktif diterapkan ke SELURUH BACKGROUND.
   3. Reset          - Buka telapak tangan lebar-lebar -> kembali ke
                       kamera normal.
   4. Portal Pinch   - Sentuh jempol + kelingking untuk berganti filter
                       secara berurutan:
                       CYBER_NEON -> SEPIA -> NEGATIVE -> THERMAL -> PIXELATE
   5. Keyboard       - 1..5 pilih filter manual, SPACE cycle, r reset,
                       h toggle help, ESC/q tutup.

 Jalankan:  py -3.14 portal_filter.py
============================================================================
"""

import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

# ── Landmark constants ─────────────────────────────────────────────────
LM_WRIST       = 0
LM_THUMB_CMC   = 1
LM_THUMB_MCP   = 2
LM_THUMB_IP    = 3
LM_THUMB_TIP   = 4
LM_INDEX_MCP   = 5
LM_INDEX_PIP   = 6
LM_INDEX_IP    = 7
LM_INDEX_TIP   = 8
LM_MIDDLE_MCP  = 9
LM_MIDDLE_PIP  = 10
LM_MIDDLE_IP   = 11
LM_MIDDLE_TIP  = 12
LM_RING_MCP    = 13
LM_RING_PIP    = 14
LM_RING_IP     = 15
LM_RING_TIP    = 16
LM_PINKY_MCP   = 17
LM_PINKY_PIP   = 18
LM_PINKY_IP    = 19
LM_PINKY_TIP   = 20

# ── Filter names ───────────────────────────────────────────────────────
FILTER_CYBER_NEON = "CYBER_NEON"
FILTER_SEPIA      = "SEPIA"
FILTER_NEGATIVE   = "NEGATIVE"
FILTER_THERMAL    = "THERMAL"
FILTER_PIXELATE   = "PIXELATE"
ALL_FILTERS = [FILTER_CYBER_NEON, FILTER_SEPIA, FILTER_NEGATIVE,
               FILTER_THERMAL, FILTER_PIXELATE]

# ── Tuning ──────────────────────────────────────────────────────────────
MAX_HANDS          = 2
PINCH_THUMB_INDEX  = 0.06     # pinch jempol-telunjuk (normalized)
PINCH_THUMB_PINKY  = 0.07     # pinch jempol-kelingking (normalized)
SLING_PINCH_DIST   = 0.05     # slingshot trigger threshold

# Model MediaPipe HandLandmarker (diunduh otomatis bila belum ada).
MODEL_FILE = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def ensure_model(model_path):
    """Unduh model .task otomatis bila belum tersedia di direktori lokal."""
    if os.path.exists(model_path):
        return
    print(f"[INFO] Model tidak ditemukan. Mengunduh {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("[INFO] Model siap.")


# ═══════════════════════════════════════════════════════════════════════
#  PORTAL FILTER APP
# ═══════════════════════════════════════════════════════════════════════
class PortalFilterApp:
    """Hand-tracking portal-filter dengan OpenCV + MediaPipe Tasks API."""

    # ── lifecycle ───────────────────────────────────────────────────────
    def __init__(self, camera_index=0):
        self.cam_idx = camera_index
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Webcam index {camera_index} tidak terdeteksi.")

        model_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), MODEL_FILE)
        ensure_model(model_path)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        # state
        self.current_filter = FILTER_CYBER_NEON
        self.frame_idx      = 0
        self.show_help      = False

        # slingshot
        self._slingshot_armed    = False   # peace terdeteksi
        self._slingshot_pinch    = False   # pinch aktif
        self._bg_filter_active   = False   # filter bg sudah aktif

        print(f"[PortalFilter] Kamera #{camera_index} OK.")
        self._print_controls()

    def cleanup(self):
        self.cap.release()
        self._landmarker.close()

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _get_norm(landmark):
        """Return array numpy [x, y] dari landmark normalisasi."""
        return np.array([landmark.x, landmark.y], dtype=np.float64)

    # ── gesture: open palm reset ────────────────────────────────────────
    def _is_open_palm_reset(self, hand_lm_list):
        wrist = self._get_norm(hand_lm_list[LM_WRIST])

        extended = 0
        # Index
        if np.linalg.norm(self._get_norm(hand_lm_list[LM_INDEX_TIP]) - wrist) > \
           np.linalg.norm(self._get_norm(hand_lm_list[LM_INDEX_PIP]) - wrist):
            extended += 1
        # Middle
        if np.linalg.norm(self._get_norm(hand_lm_list[LM_MIDDLE_TIP]) - wrist) > \
           np.linalg.norm(self._get_norm(hand_lm_list[LM_MIDDLE_PIP]) - wrist):
            extended += 1
        # Ring
        if np.linalg.norm(self._get_norm(hand_lm_list[LM_RING_TIP]) - wrist) > \
           np.linalg.norm(self._get_norm(hand_lm_list[LM_RING_PIP]) - wrist):
            extended += 1
        # Pinky
        if np.linalg.norm(self._get_norm(hand_lm_list[LM_PINKY_TIP]) - wrist) > \
           np.linalg.norm(self._get_norm(hand_lm_list[LM_PINKY_PIP]) - wrist):
            extended += 1

        # Thumb
        thumb_tip  = self._get_norm(hand_lm_list[LM_THUMB_TIP])
        thumb_mcp  = self._get_norm(hand_lm_list[LM_THUMB_MCP])
        thumb_ext  = (np.linalg.norm(thumb_tip - wrist) >
                      np.linalg.norm(thumb_mcp - wrist) * 1.10)

        return extended >= 4 and thumb_ext

    # ── gesture: pinch (jempol + telunjuk) ──────────────────────────────
    def _is_pinch_right(self, hand_lm_list):
        d = np.linalg.norm(
            self._get_norm(hand_lm_list[LM_THUMB_TIP]) -
            self._get_norm(hand_lm_list[LM_INDEX_TIP])
        )
        return d < SLING_PINCH_DIST

    # ── gesture: portal pinch (jempol + kelingking) ─────────────────────
    def _detect_portal_pinch(self, hand_lm_list):
        d = np.linalg.norm(
            self._get_norm(hand_lm_list[LM_THUMB_TIP]) -
            self._get_norm(hand_lm_list[LM_PINKY_TIP])
        )
        if d < PINCH_THUMB_PINKY:
            idx = ALL_FILTERS.index(self.current_filter)
            self.current_filter = ALL_FILTERS[(idx + 1) % len(ALL_FILTERS)]
            print(f"[Portal] Filter -> {self.current_filter}")
            return True
        return False

    # ── gesture: peace sign (telunjuk + tengah terbuka, sisanya lipat) ──
    def _is_peace(self, hand_lm_list):
        wrist = self._get_norm(hand_lm_list[LM_WRIST])

        # Index & middle extended
        idx_ext = (np.linalg.norm(self._get_norm(hand_lm_list[LM_INDEX_TIP]) - wrist) >
                   np.linalg.norm(self._get_norm(hand_lm_list[LM_INDEX_PIP]) - wrist))
        mid_ext = (np.linalg.norm(self._get_norm(hand_lm_list[LM_MIDDLE_TIP]) - wrist) >
                   np.linalg.norm(self._get_norm(hand_lm_list[LM_MIDDLE_PIP]) - wrist))

        # Ring & pinky curled
        ring_curled = (np.linalg.norm(self._get_norm(hand_lm_list[LM_RING_TIP]) - wrist) <
                       np.linalg.norm(self._get_norm(hand_lm_list[LM_RING_PIP]) - wrist) * 1.05)
        pink_curled = (np.linalg.norm(self._get_norm(hand_lm_list[LM_PINKY_TIP]) - wrist) <
                       np.linalg.norm(self._get_norm(hand_lm_list[LM_PINKY_PIP]) - wrist) * 1.05)

        return idx_ext and mid_ext and ring_curled and pink_curled

    # ── filters ─────────────────────────────────────────────────────────
    @staticmethod
    def apply_cyber_neon(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 80, 200)
        cyan = np.zeros_like(frame, dtype=np.uint8)
        cyan[:, :, 0] = 255   # B
        cyan[:, :, 1] = 255   # G
        neon = cv2.bitwise_and(cyan, cyan, mask=edge)
        blur = cv2.GaussianBlur(neon, (0, 0), 15)
        return cv2.add(frame, blur)

    @staticmethod
    def apply_sepia(frame):
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)

    @staticmethod
    def apply_negative(frame):
        return cv2.bitwise_not(frame)

    @staticmethod
    def apply_thermal(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def apply_pixelate(frame, grid=16):
        h, w = frame.shape[:2]
        tiny = cv2.resize(frame, (max(w // grid, 1), max(h // grid, 1)),
                          interpolation=cv2.INTER_LINEAR)
        return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)

    def _apply_filter(self, frame, name):
        if   name == FILTER_CYBER_NEON: return self.apply_cyber_neon(frame)
        elif name == FILTER_SEPIA:      return self.apply_sepia(frame)
        elif name == FILTER_NEGATIVE:   return self.apply_negative(frame)
        elif name == FILTER_THERMAL:    return self.apply_thermal(frame)
        elif name == FILTER_PIXELATE:   return self.apply_pixelate(frame)
        return frame

    # ── portal mask (4-corner polygon) ──────────────────────────────────
    def _build_portal_mask(self, h, w, lmL, lmR):
        """
        Build polygon mask dari 4 titik ujung jari:
        Jempol Kiri, Telunjuk Kiri, Telunjuk Kanan, Jempol Kanan.
        lmL/lmR adalah list NormalizedLandmark (indexable by int).
        """
        pts = np.array([
            [int(lmL[LM_THUMB_TIP].x * w), int(lmL[LM_THUMB_TIP].y * h)],
            [int(lmL[LM_INDEX_TIP].x * w), int(lmL[LM_INDEX_TIP].y * h)],
            [int(lmR[LM_INDEX_TIP].x * w), int(lmR[LM_INDEX_TIP].y * h)],
            [int(lmR[LM_THUMB_TIP].x * w), int(lmR[LM_THUMB_TIP].y * h)],
        ], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        return mask, pts

    # ── process frame ───────────────────────────────────────────────────
    def process(self, frame):
        h, w = frame.shape[:2]

        # Tasks API expects mp.Image + timestamp_ms.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms = int(time.perf_counter() * 1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_img, ts_ms)

        # hands_data: list of list-of-NormalizedLandmark (one per detected hand)
        hands_data = []
        if result is not None and result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                hands_data.append(hand_lm)

        # ── 1. Portal pinch -> cycle filter ─────────────────────────────
        for hand_lm in hands_data:
            self._detect_portal_pinch(hand_lm)

        # ── 2. Reset: open palm ─────────────────────────────────────────
        reset_detected = False
        for hand_lm in hands_data:
            if self._is_open_palm_reset(hand_lm):
                reset_detected = True
                break

        if reset_detected and self._bg_filter_active:
            self._bg_filter_active = False
            self._slingshot_armed  = False
            print("[Portal] RESET -> kamera normal.")

        # ── 3. Slingshot logic ──────────────────────────────────────────
        peace_left = False
        pinch_right = False
        if len(hands_data) >= 2:
            peace_left  = self._is_peace(hands_data[0])
            pinch_right = self._is_pinch_right(hands_data[1])

        if peace_left:
            self._slingshot_armed = True
        if self._slingshot_armed and not self._bg_filter_active:
            self._slingshot_pinch = pinch_right
            if self._slingshot_armed and not pinch_right and self._slingshot_pinch:
                # Release detected -> fire!
                self._bg_filter_active = True
                print(f"[Portal] SLINGSHOT FIRE -> {self.current_filter}")
        if not peace_left:
            self._slingshot_armed = False
            self._slingshot_pinch = False

        # ── 4. Portal frame (2 hands) ──────────────────────────────────
        output = frame.copy()

        if len(hands_data) >= 2:
            mask, pts = self._build_portal_mask(h, w, hands_data[0], hands_data[1])

            filtered = self._apply_filter(frame, self.current_filter)
            mask3 = cv2.merge([mask, mask, mask])
            portal_region = cv2.bitwise_and(filtered, mask3)
            bg_region     = cv2.bitwise_and(output, cv2.bitwise_not(mask3))
            output = cv2.add(portal_region, bg_region)

            # Gambar garis portal
            cv2.polylines(output, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            # Label filter di tengah portal
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(output, self.current_filter, (cx - 60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        cv2.LINE_AA)

        # ── 5. Background filter (slingshot) ────────────────────────────
        if self._bg_filter_active:
            output = self._apply_filter(frame, self.current_filter)
            cv2.putText(output, f"BG: {self.current_filter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        cv2.LINE_AA)

        # ── 6. HUD overlay ──────────────────────────────────────────────
        status = f"Filter: {self.current_filter}"
        cv2.putText(output, status, (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                    cv2.LINE_AA)

        gesture_info = ""
        if peace_left:
            gesture_info = "Peace (L) detected"
        if pinch_right:
            gesture_info = "Pinch (R) detected"
        if self._slingshot_armed and not self._bg_filter_active:
            gesture_info = "Slingshot ARMED - release to fire"
        if gesture_info:
            cv2.putText(output, gesture_info, (10, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                        cv2.LINE_AA)

        if reset_detected:
            cv2.putText(output, "RESET", (w // 2 - 40, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                        cv2.LINE_AA)

        return output

    # ── UI text ─────────────────────────────────────────────────────────
    def _print_controls(self):
        print("-" * 56)
        print("  KONTROL")
        print("-" * 56)
        print("  1-5       Pilih filter langsung")
        print("  SPACE     Cycle filter")
        print("  r         Reset background")
        print("  h         Toggle help overlay")
        print("  q / ESC   Keluar")
        print("-" * 56)
        print("  GESTURE")
        print("-" * 56)
        print("  Peace (L) + Pinch (R) -> Slingshot ARM")
        print("  Release pinch         -> FIRE filter ke BG")
        print("  Open palm             -> RESET ke normal")
        print("  Thumb + Pinky pinch   -> Cycle filter")
        print("-" * 56)

    # ── main loop ───────────────────────────────────────────────────────
    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("[PortalFilter] Frame gagal, keluar.")
                    break
                frame = cv2.flip(frame, 1)

                key = cv2.waitKey(1) & 0xFF

                # Keyboard controls
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    self._bg_filter_active = False
                    self._slingshot_armed  = False
                    print("[Portal] Manual reset -> normal.")
                elif key == ord("h"):
                    self.show_help = not self.show_help
                elif key == ord(" "):
                    idx = ALL_FILTERS.index(self.current_filter)
                    self.current_filter = ALL_FILTERS[(idx + 1) % len(ALL_FILTERS)]
                    print(f"[Portal] Filter -> {self.current_filter}")
                elif ord("1") <= key <= ord("5"):
                    self.current_filter = ALL_FILTERS[key - ord("1")]
                    print(f"[Portal] Filter -> {self.current_filter}")

                output = self.process(frame)

                if self.show_help:
                    cv2.putText(output, "1-5:filter  SPACE:cycle  r:reset  h:help  q:quit",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow("Portal Filter", output)
                self.frame_idx += 1

        finally:
            self.cleanup()
            cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
def choose_camera():
    """Tampilkan daftar kamera & minta user pilih index."""
    print("\n=== Portal Filter ===\n")
    print("Memindai kamera...")

    available = []
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, test = cap.read()
            if ok and test is not None:
                h, w = test.shape[:2]
                available.append((idx, w, h))
                print(f"  [{idx}] Kamera {idx}  ({w}x{h})")
            cap.release()

    if not available:
        print("[ERROR] Tidak ada kamera ditemukan.")
        sys.exit(1)

    if len(available) == 1:
        print(f"\nHanya 1 kamera -> otomatis pakai index {available[0][0]}.")
        return available[0][0]

    while True:
        try:
            choice = int(input(f"\nPilih index kamera [0..{len(available)-1}]: "))
            if any(idx == choice for idx, *_ in available):
                return choice
            print("Index tidak valid.")
        except ValueError:
            print("Masukkan angka.")


if __name__ == "__main__":
    cam_idx = choose_camera()
    app = PortalFilterApp(camera_index=cam_idx)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.cleanup()
        cv2.destroyAllWindows()
