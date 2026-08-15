"""
============================================================================
 Neon Hand Tracker - Sci-Fi HUD / Cyberpunk UI
============================================================================
 Deteksi tangan via webcam (MediaPipe HandLandmarker / Tasks API) dengan
 visual HUD cyberpunk yang kaya geometri, semua digambar sendiri dari nol
 (tanpa mp_draw):

  1. Kabel Laser Multi-Warna (Multi-Color Laser Cables) - saat 2 tangan
     terdeteksi, tiap pasang ujung jari kiri->kanan dihubungkan dengan
     warna neon berbeda (kuning, merah, cyan, hijau, magenta)
     lengkap dengan glow masing-masing.
  2. Trigger Kontak (Menghubungkan Sirkuit) - saat telunjuk ATAU jempol
     kiri-kanan < 40 px: background di antara kedua tangan di-blur dinamis
     (ROI, tangan tetap tajam), radial burst putih di titik kontak, dan
     semua kabel laser berdenyut lebih tebal & terang.
  3. Busur Sudut - cv2.ellipse di persendian jempol (3) & telunjuk (7)
     dengan nilai sudut (mis. "45.2 deg") di samping busur.
  4. Glow Asli - draw_glow_circle / draw_glow_line dengan GaussianBlur
     + blending overlay untuk pendaran neon cyan.
   5. Koordinat samar ukuran sangat kecil di titik-titik utama (efek
      'data scanning').
   6. Prisma Lensa Multi-Zona (Multi-Zone Prism Lens) - 10 ujung jari
      (5 kiri + 5 kanan) membentuk 4 zona segi empat yang bersambungan
      (Jempol-Telunjuk, Telunjuk-Tengah, Tengah-Manis, Manis-Kelingking).
      Setiap zona di-warp perspektif (cv2.getPerspectiveTransform /
      cv2.warpPerspective) lalu diberi ColorMap spektrum BERBEDA:
      Z1 COLORMAP_JET (thermal), Z2 COLORMAP_OCEAN (cyber blue), Z3
      COLORMAP_SUMMER (matrix green), Z4 COLORMAP_MAGMA (synthwave).
      Border & bracket sudut berwarna khas tiap pasangan jari
      (kuning/merah/cyan/hijau/magenta) + tepi glow cyan redup.

 Warna utama (BGR): Cyan Neon (255, 255, 0) & laser multi-warna.

 Kebutuhan library (install via pip):
     pip install opencv-python mediapipe numpy

 Catatan: MediaPipe 1.x memakai Tasks API. Model 'hand_landmarker.task'
 diunduh otomatis bila belum ada di folder ini.

 Cara menjalankan:  python CompVi.py
 Kontrol:           'q' tutup, 'e' aktif/mati lensa prisma, 'f' tampilkan/
                    sembunyikan bingkai & bracket.
============================================================================
"""

import math
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
# Landmark utama (MediaPipe HandLandmarker 0..20).
# ujung jempol, telunjuk, tengah, manis, kelingking
FINGER_IDS = [4, 8, 12, 16, 20]
PALM = 9                    # pusat telapak (MCP tengah)
THUMB_JOINT = 3                   # sendi IP jempol (busur sudut)
INDEX_JOINT = 7                   # sendi DIP telunjuk (busur sudut)

# Palet warna neon (format BGR karena OpenCV).
NEON_CYAN = (255, 255, 0)       # cyan neon (utama)
NEON_BONE = (150, 170, 80)      # cyan redup: kerangka glove
COORD_TEXT = (120, 150, 80)      # teks koordinat samar (data scanning)

# Warna laser per pasang ujung jari (kiri -> kanan) saat 2 tangan terdeteksi.
FINGER_LASER = {
    4:  (0, 255, 255),    # Jempol -> kuning
    8:  (0, 0, 255),      # Telunjuk -> merah
    12: (255, 255, 0),    # Jari tengah -> cyan
    16: (0, 255, 0),      # Jari manis -> hijau
    20: (255, 0, 255),    # Kelingking -> magenta
}

FONT = cv2.FONT_HERSHEY_SIMPLEX

MAX_HANDS = 2                     # jumlah tangan maksimum yang dilacak

# Trigger 'menghubungkan sirkuit': jarak telunjuk-kiri->kanan ATAU jempol.
CONTACT_DIST = 40                 # ambang (px) untuk memicu efek blur energi
BLUR_SIGMA = 9                  # kekuatan GaussianBlur saat kontak
ROI_MARGIN = 40                 # bingkai tambahan di sekitar tangan (ROI)
SHARP_RADIUS = 30                 # radius area tangan yang tetap tajam

# --- Prisma Lensa Multi-Zona (5 jari x 2 tangan) -----------------------
# 4 zona segi empat bersambung: tiap zona dibatasi 2 pasang ujung jari
# bersebelahan (kiri & kanan), dengan efek spektrum (ColorMap) berbeda.
ZONE_SPECS = [
    (4, 8, cv2.COLORMAP_JET),       # Zona 1: Jempol-Telunjuk -> Thermal
    (8, 12, cv2.COLORMAP_OCEAN),    # Zona 2: Telunjuk-Tengah -> Cyber Blue
    (12, 16, cv2.COLORMAP_SUMMER),  # Zona 3: Tengah-Manis -> Matrix Green
    (16, 20, cv2.COLORMAP_MAGMA),   # Zona 4: Manis-Kelingking -> Synthwave
]
GLASS_WARP_SIZE = 320            # ukuran target warp perspektif per zona (px)
GLASS_MIN_AREA = 120             # quad lebih kecil -> lewati (FPS stabil)
GLASS_SPEC_ALPHA = 0.65          # kekuatan ColorMap (alpha blending)
GLASS_TINT_ALPHA = 0.15          # kilau kaca seragam (alpha, semua zona)
FRAME_COLOR = (255, 255, 255)    # putih bersih: tepi sisi zona
CORNER_LEN = 22                  # panjang lengan bracket L (px)
ZONE_EDGE = (160, 170, 90)       # cyan redup: sisi vertikal antar zona

# Model MediaPipe HandLandmarker (diunduh otomatis bila belum ada).
MODEL_FILE = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# Kerangka tangan resmi dari MediaPipe: daftar pasangan landmark yang terhubung.
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


def ensure_model(model_path):
    """Unduh model .task otomatis bila belum tersedia di direktori lokal."""
    if os.path.exists(model_path):
        return
    print(f"[INFO] Model tidak ditemukan. Mengunduh {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("[INFO] Model siap.")


# ---------------------------------------------------------------------------
# Glow: gambar elemen ke layer 'bloom', lalu GaussianBlur + blending overlay.
# ---------------------------------------------------------------------------
def draw_glow_line(canvas, bloom, a, b, color, thickness=1, glow=4):
    """Garis tipis krisp di canvas + sapuan tebal di layer bloom untuk glow."""
    cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
    cv2.line(bloom, a, b, color, glow)


def draw_glow_circle(canvas, bloom, p, r, color, filled=True):
    """Lingkaran berpendaran: inti di canvas + sapuan di layer bloom."""
    if filled:
        cv2.circle(canvas, p, r, color, -1)
        cv2.circle(bloom, p, r + 2, color, -1)
    else:
        cv2.circle(canvas, p, r, color, 1, cv2.LINE_AA)
        cv2.circle(bloom, p, r, color, 3)


def apply_bloom(canvas, bloom):
    """Blur layer bloom lalu saturasi-tambahkan ke canvas untuk efek neon."""
    if not bloom.any():
        return canvas
    bloom_blur = cv2.GaussianBlur(bloom, (0, 0), 5)
    return cv2.addWeighted(canvas, 1.0, bloom_blur, 0.9, 0)


# ---------------------------------------------------------------------------
# Elemen visual per tangan
# ---------------------------------------------------------------------------
def draw_neon_glove(canvas, bloom, hand_pts):
    """Kerangka tangan (21 tulang) digambar sendiri: garis tipis cyan redup."""
    for conn in HAND_CONNECTIONS:
        pa, pb = hand_pts[conn.start], hand_pts[conn.end]
        draw_glow_line(canvas, bloom, pa, pb, NEON_BONE, 1, 3)

    # Sendi sebagai node kecil (ujung jari digambar nodes energinya sendiri).
    for i, p in enumerate(hand_pts):
        if i not in FINGER_IDS:
            draw_glow_circle(canvas, bloom, p, 2, NEON_BONE)
    # Pusat telapak sebagai titik data.
    draw_glow_circle(canvas, bloom, hand_pts[PALM], 3, NEON_CYAN, filled=False)


def draw_angle_arc(canvas, bloom, center, p_prev, p_next, radius):
    """
    Busur derajat di persendian: cv2.ellipse antara arah dua ruas tulang.
    Menampilkan nilai sudut tekukan (derajat) di samping busur.
    """
    v1 = (p_prev[0] - center[0], p_prev[1] - center[1])
    v2 = (p_next[0] - center[0], p_next[1] - center[1])
    l1 = float(np.hypot(*v1))
    l2 = float(np.hypot(*v2))
    if l1 < 1e-3 or l2 < 1e-3:
        return

    # Sudut tekukan antar dua ruas (0..180 derajat).
    cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
    bend = math.degrees(math.acos(float(np.clip(cosv, -1.0, 1.0))))

    # Arah kedua ruas, gambar busur (swipe terpendek antar arah).
    a1 = math.degrees(math.atan2(v1[1], v1[0]))
    a2 = math.degrees(math.atan2(v2[1], v2[0]))
    delta = (a2 - a1 + 180.0) % 360.0 - 180.0
    cv2.ellipse(canvas, center, (radius, radius), 0,
                a1, a1 + delta, NEON_CYAN, 1, cv2.LINE_AA)
    cv2.ellipse(bloom, center, (radius, radius), 0,
                a1, a1 + delta, NEON_CYAN, 3)

    # Nilai sudut di samping busur, font kecil & bersih.
    tx, ty = center[0] + radius + 8, center[1] + 6
    cv2.putText(canvas, f"{bend:.1f} deg", (tx, ty), FONT,
                0.4, NEON_CYAN, 1, cv2.LINE_AA)


def draw_energy_node(canvas, bloom, p, color, t_now, phase, r=3):
    """Node energi ujung jari: inti terang + halo berdenyut + glow."""
    pulse = 0.5 + 0.5 * math.sin(t_now * 6.0 + phase)
    halo = int(r + 4 + 4 * pulse)
    draw_glow_circle(canvas, bloom, p, halo, color, filled=False)
    cv2.circle(canvas, p, r + 1, color, -1)
    cv2.circle(canvas, p, r, (255, 255, 255), 1)
    cv2.circle(bloom, p, r + 2, color, -1)


def draw_tiny_coord(canvas, p):
    """Angka koordinat sangat kecil & samar (efek data scanning)."""
    cv2.putText(canvas, f"{p[0]},{p[1]}", (p[0] + 8, p[1] + 4),
                FONT, 0.3, COORD_TEXT, 1, cv2.LINE_AA)


def draw_hand_hud(canvas, bloom, hand_pts, particles, t_now):
    """Kumpulan elemen HUD untuk satu tangan."""
    draw_neon_glove(canvas, bloom, hand_pts)
    draw_angle_arc(canvas, bloom, hand_pts[THUMB_JOINT],
                   hand_pts[THUMB_JOINT - 1], hand_pts[THUMB_JOINT + 1], 18)
    draw_angle_arc(canvas, bloom, hand_pts[INDEX_JOINT],
                   hand_pts[INDEX_JOINT - 1], hand_pts[INDEX_JOINT + 1], 16)

    for idx, fid in enumerate(FINGER_IDS):
        p = hand_pts[fid]
        draw_energy_node(canvas, bloom, p, NEON_CYAN, t_now, idx * 1.1)
        draw_tiny_coord(canvas, p)
        particles.emit_trail(p[0], p[1])


# ---------------------------------------------------------------------------
# Jaring Inter-Tangan (Multi-Color Laser Cables)
# ---------------------------------------------------------------------------
def draw_inter_hand_mesh(canvas, bloom, hands, contact_pulse=0.0):
    """
    Hubungkan tiap ujung jari kiri -> kanan dengan warna laser neon tersendiri
    (FINGER_LASER). Saat kontak aktif (contact_pulse > 0), semua garis
    berdenyut lebih tebal & lebih terang.
    """
    if len(hands) < 2:
        return
    h0, h1 = hands[0], hands[1]

    # Jarak antar tangan (dari pusat telapak) untuk meredupkan garis saat jauh.
    d = float(np.hypot(h1[PALM][0] - h0[PALM][0], h1[PALM][1] - h0[PALM][1]))
    intensity = float(np.clip(1.0 - d / 320.0, 0.15, 1.0))
    if contact_pulse > 0.0:
        intensity = max(intensity, 0.9)

    thickness = 1 + int(2 * contact_pulse)
    glow = 4 + int(8 * contact_pulse)

    for fid in FINGER_IDS:
        color = tuple(int(ch * intensity) for ch in FINGER_LASER[fid])
        draw_glow_line(canvas, bloom, h0[fid], h1[fid], color, thickness, glow)


# ---------------------------------------------------------------------------
# Trigger kontak: blur energi & radial burst (menghubungkan sirkuit)
# ---------------------------------------------------------------------------
def detect_contact(h0, h1, threshold=CONTACT_DIST):
    """
    Deteksi titik pertemuan: jarak telunjuk-kiri->kanan ATAU jempol-kiri->kanan
    kurang dari 'threshold'. Mengembalikan titik tengah pasangan terdekat,
    atau None bila tidak ada kontak.
    """
    best = None
    best_d = threshold
    for fid in (4, 8):
        d = float(np.hypot(h1[fid][0] - h0[fid][0], h1[fid][1] - h0[fid][1]))
        if d < best_d:
            best_d = d
            best = ((h0[fid][0] + h1[fid][0]) // 2,
                    (h0[fid][1] + h1[fid][1]) // 2)
    return best


def apply_contact_blur(frame, canvas, hands):
    """
    Blur dinamis pada area di antara kedua tangan (ROI), lalu kembalikan area
    tangan agar tetap tajam. Hanya ROI yang diproses -> FPS stabil.
    """
    h, w = canvas.shape[:2]

    # Bounding box seluruh landmark kedua tangan + margin.
    xs = [p[0] for pts in hands for p in pts]
    ys = [p[1] for pts in hands for p in pts]
    x0, y0 = max(min(xs) - ROI_MARGIN, 0), max(min(ys) - ROI_MARGIN, 0)
    x1, y1 = min(max(xs) + ROI_MARGIN, w), min(max(ys) + ROI_MARGIN, h)
    if x1 <= x0 or y1 <= y0:
        return canvas

    # Blur hanya ROI (hemat komputasi).
    roi = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (0, 0), BLUR_SIGMA)

    # Pulihkan area tangan (video asli) agar tetap tajam/kontras.
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for pts in hands:
        for (x, y) in pts:
            cv2.circle(mask, (x - x0, y - y0), SHARP_RADIUS, 255, -1)
    roi_view = canvas[y0:y1, x0:x1]
    roi_view[mask > 0] = frame[y0:y1, x0:x1][mask > 0]
    return canvas


def draw_radial_burst(canvas, bloom, center, t_now, t):
    """
    Kilatan pendaran (radial burst) di titik pertemuan jari yang bersentuhan.
    Radius berdenyut terhadap waktu; gambar di layer bloom agar berpendar.
    """
    pulse = 0.5 + 0.5 * math.sin(t_now * 10.0)
    radius = 8 + int(22 * t * (0.6 + 0.4 * pulse))

    # Inti putih panas di pusat kontak.
    cv2.circle(canvas, center, 4, (255, 255, 255), -1)
    cv2.circle(bloom, center, radius, (255, 255, 255), -1)

    # Garis-garis radial sebagai semburan energi.
    for ang in range(0, 360, 30):
        rad = math.radians(ang)
        end = (int(center[0] + (radius + 10) * math.cos(rad)),
               int(center[1] + (radius + 10) * math.sin(rad)))
        cv2.line(canvas, center, end, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(bloom, center, end, (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# Bingkai Kaca Virtual (Virtual Glass Frame) & Keyframe Corners
# ---------------------------------------------------------------------------
def _unit_dir(a, b):
    """Vektor satuan dari a menuju b (untuk arah lengan bracket L)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy)
    if d < 1e-3:
        return (0.0, 0.0)
    return (dx / d, dy / d)


def order_quad_corners(points):
    """
    Urutkan 4 titik menjadi segi empat siklik searah jarum jam di sekitar
    centroid. Menjamin sisi bingkai selalu antar sudut yang berdekatan,
    sehingga polygon & perspective transform valid (tidak menyilang).
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return np.array(pts, dtype=np.float32)


def build_zone_quad(h0, h1, fid_top, fid_bot):
    """
    Segi empat satu zona lensa prisma: 4 titik dari 2 pasang ujung jari
    bersebelahan (kiri & kanan), diurutkan siklik agar warp tetap valid.
    """
    return order_quad_corners([h0[fid_top], h0[fid_bot],
                               h1[fid_top], h1[fid_bot]])


def build_zone_quads(h0, h1):
    """Keempat zona lensa prisma (Z1..Z4) dari 10 ujung jari kedua tangan."""
    return [build_zone_quad(h0, h1, top, bot) for top, bot, _ in ZONE_SPECS]


def _make_glass_overlay(width, height):
    """
    Overlay kilau kaca halus (gradasi cyan -> ungu + pantulan diagonal)
    yang dipakai seragam di seluruh zona sebagai 'permukaan prisma'.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = np.clip((xx + yy) / max(width + height - 2, 1) * 1.5, 0.0, 1.0)
    cyan = np.array([255, 235, 0], np.float32)      # BGR
    purple = np.array([190, 70, 220], np.float32)   # BGR
    overlay = np.zeros((height, width, 3), np.float32)
    for c in range(3):
        overlay[:, :, c] = cyan[c] * (1.0 - t) + purple[c] * t
    shimmer = ((xx + yy).astype(np.int32) % 80) < 2   # pantulan diagonal
    overlay[shimmer] = np.minimum(overlay[shimmer] + 40.0, 255.0)
    return overlay.astype(np.uint8)


# Cache overlay kaca (ukuran warp konstan) -> tidak dihitung ulang tiap frame.
_GLASS_OVERLAY_CACHE = {}


def _get_glass_overlay(size):
    if size not in _GLASS_OVERLAY_CACHE:
        _GLASS_OVERLAY_CACHE[size] = _make_glass_overlay(size, size)
    return _GLASS_OVERLAY_CACHE[size]


def render_zone_lens(frame, canvas, quad, colormap):
    """
    Warp ROI satu zona ke bidang perspektif netral
    (cv2.getPerspectiveTransform + warpPerspective), terapkan ColorMap
    spektrum khas zona + kilau kaca, lalu komposit balik HANYA di dalam
    segi empat. Aman: koordinat dijepit layar, quad degenerasi dilewati
    -> tidak crash & FPS tetap stabil.
    """
    h, w = frame.shape[:2]
    pts = np.asarray(quad, np.float32)

    # Keamanan: semua sudut harus berada di dalam layar.
    if np.any(pts[:, 0] < 0) or np.any(pts[:, 1] < 0) or \
       np.any(pts[:, 0] >= w) or np.any(pts[:, 1] >= h):
        return

    # Quad terlalu kecil / menyilang -> lewati (warp tidak stabil).
    if cv2.contourArea(pts) < GLASS_MIN_AREA:
        return

    # ROI lokal (dijepit ke area layar).
    x0 = max(int(pts[:, 0].min()) - 2, 0)
    y0 = max(int(pts[:, 1].min()) - 2, 0)
    x1 = min(int(pts[:, 0].max()) + 2, w)
    y1 = min(int(pts[:, 1].max()) + 2, h)
    rw, rh = x1 - x0, y1 - y0
    if rw < 8 or rh < 8:
        return

    pts_local = pts - np.array([x0, y0], np.float32)
    dst = np.array([[0, 0],
                    [GLASS_WARP_SIZE - 1, 0],
                    [GLASS_WARP_SIZE - 1, GLASS_WARP_SIZE - 1],
                    [0, GLASS_WARP_SIZE - 1]], np.float32)

    # Warp depan: isi zona 'menghadap kamera' (perspective crop).
    m_fwd = cv2.getPerspectiveTransform(pts_local, dst)
    roi = frame[y0:y1, x0:x1]
    quad_img = cv2.warpPerspective(roi, m_fwd,
                                   (GLASS_WARP_SIZE, GLASS_WARP_SIZE))

    # Efek spektrum zona: ColorMap khas + kilau kaca seragam.
    gray = cv2.cvtColor(quad_img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, colormap)
    quad_eff = cv2.addWeighted(quad_img, 1.0 - GLASS_SPEC_ALPHA,
                               colored, GLASS_SPEC_ALPHA, 0)
    if GLASS_TINT_ALPHA > 0.0:
        overlay = _get_glass_overlay(GLASS_WARP_SIZE)
        quad_eff = cv2.addWeighted(quad_eff, 1.0 - GLASS_TINT_ALPHA,
                                   overlay, GLASS_TINT_ALPHA, 0)
    cv2.rectangle(quad_eff, (1, 1),
                  (GLASS_WARP_SIZE - 2, GLASS_WARP_SIZE - 2),
                  (255, 255, 255), 1)

    # Warp balik ke posisi asli (koordinat ROI).
    m_bwd = cv2.getPerspectiveTransform(dst, pts_local)
    warped = cv2.warpPerspective(quad_eff, m_bwd, (rw, rh))

    # Masker hanya area dalam quad; luar zona tetap video asli.
    mask = np.zeros((rh, rw), np.uint8)
    cv2.fillPoly(mask, [pts_local.astype(np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    roi_view = canvas[y0:y1, x0:x1]
    roi_view[mask > 0] = warped[mask > 0]


def render_prism_lens(frame, canvas, quads):
    """Terapkan efek spektrum ke keempat zona lensa secara simultan."""
    for quad, (_, _, colormap) in zip(quads, ZONE_SPECS):
        render_zone_lens(frame, canvas, quad, colormap)


def _sort_corners_tagged(corners):
    """
    Urutkan 4 pasangan (titik, label) menjadi segi empat siklik di sekitar
    centroid; label tetap terbawa agar warna bracket cocok dengan jarinya.
    """
    pts = [(float(p[0]), float(p[1])) for p, _ in corners]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    return sorted(corners,
                  key=lambda t: math.atan2(t[0][1] - cy, t[0][0] - cx))


def draw_corner_bracket(canvas, bloom, corner, prev, nxt,
                        color=FRAME_COLOR, length=CORNER_LEN):
    """
    Bracket L (keyframe corner anchor) di satu sudut zona, berwarna khas
    pasangan jari yang bersangkutan.
    """
    dx1, dy1 = _unit_dir(corner, prev)
    dx2, dy2 = _unit_dir(corner, nxt)
    p1 = (int(corner[0] + dx1 * length), int(corner[1] + dy1 * length))
    p2 = (int(corner[0] + dx2 * length), int(corner[1] + dy2 * length))
    cv2.line(canvas, corner, p1, color, 2, cv2.LINE_AA)
    cv2.line(canvas, corner, p2, color, 2, cv2.LINE_AA)
    cv2.line(bloom, corner, p1, color, 5)
    cv2.line(bloom, corner, p2, color, 5)


def draw_zone_brackets(canvas, bloom, h0, h1):
    """
    Keyframe corner anchors: bracket L berwarna khas pasangan jari di
    sudut-sudut keempat zona (mencakup seluruh 10 ujung jari L & R).
    """
    drawn = set()
    for fid_top, fid_bot, _ in ZONE_SPECS:
        tagged = _sort_corners_tagged([
            (h0[fid_top], ("L", fid_top)),
            (h0[fid_bot], ("L", fid_bot)),
            (h1[fid_top], ("R", fid_top)),
            (h1[fid_bot], ("R", fid_bot)),
        ])
        for i in range(4):
            p, (hand, fid) = tagged[i]
            if (hand, fid) in drawn:
                continue
            drawn.add((hand, fid))
            prev = tagged[(i - 1) % 4][0]
            nxt = tagged[(i + 1) % 4][0]
            draw_corner_bracket(canvas, bloom, p, prev, nxt,
                                color=FINGER_LASER[fid], length=CORNER_LEN)


def draw_zone_edges(canvas, bloom, h0, h1):
    """
    Sisi vertikal tiap zona (antar ujung jari bersebelahan): garis tipis
    glow cyan redup sebagai pembatas sel zona.
    """
    for i in range(len(FINGER_IDS) - 1):
        a, b = FINGER_IDS[i], FINGER_IDS[i + 1]
        draw_glow_line(canvas, bloom, h0[a], h0[b], ZONE_EDGE, 1, 3)
        draw_glow_line(canvas, bloom, h1[a], h1[b], ZONE_EDGE, 1, 3)


def draw_prism_frame(canvas, bloom, h0, h1):
    """
    Bingkai prisma lensa: sisi vertikal zona + bracket L warna khas di
    10 ujung jari (garis horizontal pasangan jari digambar laser mesh).
    """
    draw_zone_edges(canvas, bloom, h0, h1)
    draw_zone_brackets(canvas, bloom, h0, h1)


# ---------------------------------------------------------------------------
# Partikel jejak
# ---------------------------------------------------------------------------
class ParticleSystem:
    """Sistem jejak partikel neon yang membuntuti ujung jari."""

    def __init__(self):
        self.particles = []

    def _add(self, x, y, vx, vy, life, size, color):
        self.particles.append({
            "x": x, "y": y, "vx": vx, "vy": vy,
            "life": life, "max_life": life, "size": size, "color": color,
        })

    def emit_trail(self, x, y, n=2):
        """Keluar partikel kecil yang membuntuti posisi ujung jari."""
        for _ in range(n):
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            spd = np.random.uniform(15, 60)
            self._add(
                x + np.random.uniform(-4, 4), y + np.random.uniform(-4, 4),
                np.cos(angle) * spd, np.sin(angle) * spd,
                np.random.uniform(0.25, 0.6), np.random.uniform(1.5, 3.5),
                NEON_CYAN,
            )

    def update(self, dt):
        """Integrasi posisi, redam kecepatan, matikan partikel yang sudah mati."""
        for p in self.particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["vx"] *= 0.90
            p["vy"] *= 0.90
            p["life"] -= dt
        self.particles = [p for p in self.particles if p["life"] > 0]

    def draw(self, canvas):
        """Gambar partikel transparan (halo) + inti terang di atas canvas."""
        overlay = canvas.copy()
        for p in self.particles:
            t = max(p["life"] / p["max_life"], 0.0)
            x, y = int(p["x"]), int(p["y"])
            r = int(p["size"] * (0.4 + 0.6 * t)) + 1
            halo = tuple(int(ch * t) for ch in p["color"])
            cv2.circle(overlay, (x, y), r + 2, halo, -1)
            cv2.circle(canvas, (x, y), max(r - 1, 1), p["color"], -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)


def main():
    # -----------------------------------------------------------------------
    # Setup kamera & MediaPipe HandLandmarker (Tasks API).
    # -----------------------------------------------------------------------
    # CAP_DSHOW: backend DirectShow Windows — lebih andal dari MSMF default
    # di mesin ini (MSMF terbuka tapi tidak bisa mengambil frame).
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka webcam (index 0).")
        return

    model_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), MODEL_FILE)
    ensure_model(model_path)

    # RunningMode.VIDEO: memproses tiap frame dengan timestamp yang menaik.
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    particles = ParticleSystem()
    prev_time = time.perf_counter()
    prism_on = True
    show_frame = True

    try:
        with landmarker:
            while True:
                # Delta waktu frame (dt) untuk animasi partikel.
                now = time.perf_counter()
                dt = max(now - prev_time, 1e-6)
                prev_time = now

                success, frame = cap.read()
                if not success:
                    print("[INFO] Feed webcam terputus.")
                    break

                # Mirror horizontal agar gerakan terasa natural (seperti cermin).
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # Canvas = feed video; layer bloom terpisah untuk efek glow.
                canvas = frame.copy()
                bloom = np.zeros_like(canvas)

                # Frame harus berupa RGB dan timestamp menaik (ms).
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(time.perf_counter() * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                hands = []
                if result is not None and result.hand_landmarks:
                    # Ambil koordinat (px) seluruh 21 landmark tiap tangan.
                    for hand_lm in result.hand_landmarks:
                        hands.append([(int(lm.x * w), int(lm.y * h))
                                     for lm in hand_lm])

                t_now = time.perf_counter()

                # Trigger kontak: telunjuk ATAU jempol kiri-kanan saling dekat.
                contact = None
                pulse = 0.0
                zone_quads = None
                if len(hands) >= 2:
                    contact = detect_contact(hands[0], hands[1], CONTACT_DIST)
                    # 4 zona lensa prisma dari 10 ujung jari (5 L + 5 R).
                    zone_quads = build_zone_quads(hands[0], hands[1])
                if contact is not None:
                    # Blur energi di antara kedua tangan; tangan tetap tajam.
                    canvas = apply_contact_blur(frame, canvas, hands)
                    pulse = 0.5 + 0.5 * math.sin(t_now * 10.0)

                # Lensa prisma: efek spektrum berbeda per zona (ROI + warp).
                if zone_quads is not None and prism_on:
                    render_prism_lens(frame, canvas, zone_quads)

                for hand_pts in hands:
                    draw_hand_hud(canvas, bloom, hand_pts, particles, t_now)

                if len(hands) >= 2:
                    # Kabel laser multi-warna; berdenyut saat kontak aktif.
                    draw_inter_hand_mesh(canvas, bloom, hands, pulse)

                if contact is not None:
                    draw_radial_burst(canvas, bloom, contact, t_now, pulse)

                # Bingkai prisma & bracket warna khas (di atas semua HUD).
                if len(hands) >= 2 and show_frame:
                    draw_prism_frame(canvas, bloom, hands[0], hands[1])

                # Blur & saturasi-tambahkan layer glow ke canvas.
                canvas = apply_bloom(canvas, bloom)

                # Gerakkan & gambar partikel jejak.
                particles.update(dt)
                particles.draw(canvas)

                cv2.imshow("Neon Hand - Sci-Fi HUD", canvas)

                # Kontrol: 'q' tutup, 'e' lensa prisma on/off, 'f' bingkai.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("e"):
                    prism_on = not prism_on
                    print(f"[INFO] Lensa prisma: {'ON' if prism_on else 'OFF'}")
                elif key == ord("f"):
                    show_frame = not show_frame
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
