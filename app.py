from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def scroll_to_top() -> None:
    """Cuộn cửa sổ lên đầu trang (gọi sau khi bấm Định giá)."""
    components.html(
        """
        <script>
            const doc = window.parent.document;
            const el = doc.querySelector('section.main') ||
                       doc.querySelector('[data-testid="stMain"]') ||
                       doc.querySelector('.main');
            if (el) { el.scrollTo({top: 0, behavior: 'smooth'}); }
            window.parent.scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """,
        height=0,
    )

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "chungcu_ready_to_train.xlsx"
MODEL_PATH = BASE_DIR / "model_chung_cu.pkl"
FEATURES_PATH = BASE_DIR / "features_chung_cu.pkl"

# --- Nhà đất ---
NHADAT_EXCEL_PATH = BASE_DIR / "nhadat_ready_to_train.xlsx"
NHADAT_MODEL_PATH = BASE_DIR / "model_nha_dat.pkl"
NHADAT_FEATURES_PATH = BASE_DIR / "features_nha_dat.pkl"
NHADAT_ENCODERS_PATH = BASE_DIR / "encoders_nha_dat.pkl"

LEGAL_BASELINE = "Hợp đồng mua bán"
STATUS_BASELINE = "Chưa bàn giao"
PROJECT_OTHER = "__other__"   # "Dự án khác" -> ước lượng theo phường

INTERIOR_CHOICES = [
    "Nội thất đầy đủ",
    "Nội thất cao cấp",
    "Hoàn thiện cơ bản",
    "Bàn giao thô",
    "Không rõ",
]
DIRECTION_CHOICES = [
    "Đông",
    "Tây",
    "Nam",
    "Bắc",
    "Đông Bắc",
    "Tây Bắc",
    "Đông Nam",
    "Tây Nam",
    "Không rõ",
]
LEGAL_CHOICES = [
    "Sổ hồng riêng",
    "Hợp đồng mua bán",
    "Đang chờ sổ",
    "Hợp đồng đặt cọc",
    "Không rõ",
]
STATUS_CHOICES = ["Đã bàn giao", "Chưa bàn giao", "Không rõ"]


# =============================================================================
# CHUẨN HÓA TÊN DỰ ÁN  (xử lý dữ liệu đầu vào — KHÔNG đụng logic mô hình)
# Gazetteer các dự án chung cư Hà Nội: alias (chữ thường, không dấu) -> tên chuẩn.
# Khớp cụm cụ thể TRƯỚC cụm chung.
# =============================================================================
UNKNOWN_PROJECT = "Không xác định"

GAZETTEER: list[tuple[str, list[str]]] = [
    ("Vinhomes Ocean Park",    [r"vinhomes ocean park", r"\bocean park\b"]),
    ("Vinhomes Smart City",    [r"vinhomes\s*smart\s*city", r"smart\s*city", r"smartcity"]),
    ("Vinhomes Gardenia",      [r"vinhomes gardenia", r"\bgardenia\b"]),
    ("Vinhomes Green Bay",     [r"green bay"]),
    ("Times City",             [r"time?s city", r"park hill"]),
    ("Royal City",             [r"royal city"]),
    ("Lumière Evergreen",      [r"lumi[eè]re"]),
    ("Imperia Sky Garden",     [r"imperia sky garden"]),
    ("Imperia Parkland",       [r"imperia parkland"]),
    ("Imperia",                [r"\bimperia\b"]),
    ("HH Linh Đàm",            [r"hh[0-9abc]*\s+linh dam", r"hh.{0,8}linh dam", r"ban dao linh dam"]),
    ("Rice City Sông Hồng",    [r"ri[cs]e city song hong"]),
    ("Rice City Linh Đàm",     [r"rice city"]),
    ("Kim Văn Kim Lũ",         [r"kim van kim lu"]),
    ("KĐT Đại Thanh",          [r"(ct[0-9abz]* )?dai thanh"]),
    ("KĐT Linh Đàm",           [r"linh dam"]),
    ("FLC Garden City Đại Mỗ", [r"flc .*(garden|dai mo)", r"flc garden city"]),
    ("FLC",                    [r"\bflc\b"]),
    ("Ecohome Phúc Lợi",       [r"ecohome phuc loi"]),
    ("Ecohome",                [r"ecohome", r"eco home"]),
    ("Eco Green City",         [r"eco green"]),
    ("Masteri",                [r"masteri"]),
    ("Goldmark City",          [r"goldmark"]),
    ("Gold Season",            [r"gold season"]),
    ("Mipec",                  [r"mipec"]),
    ("KĐT Ngoại Giao Đoàn",    [r"ngoai giao doan", r"\bngoai giao\b"]),
    ("The Pride",              [r"the pride"]),
    ("KĐT Văn Khê",            [r"van khe"]),
    ("The Zei",                [r"the zei"]),
    ("The Sola Park",          [r"the sola"]),
    ("The Matrix One",         [r"the matrix"]),
    ("The Emerald",            [r"the emerald"]),
    ("Roman Plaza",            [r"roman plaza"]),
    ("Five Star Kim Giang",    [r"five star"]),
    ("HPC Landmark 105",       [r"hpc landmark"]),
    ("Hateco",                 [r"hateco"]),
    ("Dream Town",             [r"dream town"]),
    ("Sunshine",               [r"sunshine"]),
    ("Gemek",                  [r"gemek"]),
    ("Feliz Homes",            [r"feliz"]),
    ("TSQ Euroland",           [r"\btsq\b"]),
    ("King Palace",            [r"king palace"]),
    ("Ruby City",              [r"ruby city", r"\bruby\b"]),
    ("Ciputra",                [r"ciputra"]),
    ("Roman / Hồng Hà",        [r"hong ha"]),
    ("Usilk City",             [r"usilk"]),
    ("Gelexia Riverside",      [r"gelexia"]),
    ("Berriver Long Biên",     [r"berriver"]),
    ("Eurowindow River Park",  [r"eurowindow"]),
    ("KĐT Thành Phố Giao Lưu", [r"thanh pho giao luu", r"\btp giao luu\b"]),
    ("X2 Đại Kim",             [r"x2 dai kim"]),
    ("CT2A Thạch Bàn",         [r"ct2a thach ban"]),
]


def _strip_accents(s: str) -> str:
    # 'đ'/'Đ' KHÔNG bị NFD tách dấu -> phải thay thủ công về 'd'.
    s = s.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()


def normalize_project_name(raw: object) -> str:
    """Tiêu đề tin đăng (nhiễu) -> tên dự án chuẩn, hoặc UNKNOWN_PROJECT."""
    if raw is None:
        return UNKNOWN_PROJECT
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "-"}:
        return UNKNOWN_PROJECT
    flat = _strip_accents(text)
    for canonical, aliases in GAZETTEER:
        for pat in aliases:
            if re.search(pat, flat):
                return canonical
    return UNKNOWN_PROJECT


# =============================================================================
# DATA / MODEL LOADING — GIỮ NGUYÊN 100% (không đổi logic nạp file)
# =============================================================================
@st.cache_data(show_spinner="Đang đọc dữ liệu huấn luyện...")
def load_training_data() -> pd.DataFrame:
    return pd.read_excel(EXCEL_PATH)


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_model_bundle():
    with open(MODEL_PATH, "rb") as handle:
        model = pickle.load(handle)
    with open(FEATURES_PATH, "rb") as handle:
        features = pickle.load(handle)
    return model, features


# ---- Nhà đất: loaders + cây Quận→Phường ----
@st.cache_resource(show_spinner="Đang tải mô hình nhà đất...")
def load_nhadat_bundle():
    with open(NHADAT_MODEL_PATH, "rb") as h:
        model = pickle.load(h)
    with open(NHADAT_FEATURES_PATH, "rb") as h:
        features = pickle.load(h)
    with open(NHADAT_ENCODERS_PATH, "rb") as h:
        encoders = pickle.load(h)
    return model, features, encoders


@st.cache_data(show_spinner="Đang đọc dữ liệu nhà đất...")
def load_nhadat_hierarchy() -> dict:
    """Dựng cây Quận→Phường từ dữ liệu nhà đất (để selector phân cấp)."""
    df = pd.read_excel(NHADAT_EXCEL_PATH, usecols=["quan_huyen", "phuong_xa"])
    df = df.dropna(subset=["quan_huyen"])
    hierarchy: dict[str, list[str]] = {}
    for quan, grp in df.groupby("quan_huyen"):
        hierarchy[quan] = sorted(grp["phuong_xa"].dropna().unique())
    return {"hierarchy": hierarchy, "n_rows": len(df)}


@st.cache_data
def build_location_encodings(df: pd.DataFrame) -> dict:
    quan_map = (
        df.dropna(subset=["quan_huyen"])
        .groupby("quan_huyen")["quan_huyen_encoded"]
        .first()
        .astype(float)
        .to_dict()
    )

    phuong_map = {}
    phuong_subset = df.dropna(subset=["quan_huyen", "phuong_xa"])
    for (quan, phuong), group in phuong_subset.groupby(["quan_huyen", "phuong_xa"]):
        phuong_map[(quan, phuong)] = float(group["phuong_xa_encoded"].iloc[0])

    hierarchy: dict[str, list[str]] = {}
    for quan in sorted(quan_map.keys()):
        phuongs = sorted(
            phuong_subset.loc[phuong_subset["quan_huyen"] == quan, "phuong_xa"].unique()
        )
        hierarchy[quan] = phuongs

    # ---- CHUẨN HÓA dự án + bảng tra encoding theo dự án đã chuẩn hóa ----
    work = df.copy()
    src = work["ten_du_an"].fillna(work.get("du_an"))
    work["du_an_clean"] = src.map(normalize_project_name)
    named = work[work["du_an_clean"] != UNKNOWN_PROJECT]

    project_enc_loc: dict[tuple[str, str, str], float] = {
        (q, p, n): float(v)
        for (q, p, n), v in named.groupby(
            ["quan_huyen", "phuong_xa", "du_an_clean"]
        )["du_an_encoded"].mean().items()
    }
    project_enc_quan: dict[tuple[str, str], float] = {
        (q, n): float(v)
        for (q, n), v in named.groupby(
            ["quan_huyen", "du_an_clean"]
        )["du_an_encoded"].mean().items()
    }

    project_by_location: dict[tuple[str, str], list[str]] = {}
    for (q, p), grp in named.groupby(["quan_huyen", "phuong_xa"]):
        project_by_location[(q, p)] = sorted(grp["du_an_clean"].unique())
    project_by_quan: dict[str, list[str]] = {}
    for q, grp in named.groupby("quan_huyen"):
        project_by_quan[q] = sorted(grp["du_an_clean"].unique())

    global_mean = float(df["gia_tren_m2"].mean())
    return {
        "quan_map": quan_map,
        "phuong_map": phuong_map,
        "hierarchy": hierarchy,
        "project_enc_loc": project_enc_loc,
        "project_enc_quan": project_enc_quan,
        "project_by_location": project_by_location,
        "project_by_quan": project_by_quan,
        "n_projects": int(named["du_an_clean"].nunique()),
        "global_mean": global_mean,
    }


# =============================================================================
# FEATURE ENGINEERING — GIỮ NGUYÊN logic mô hình.
# Chỉ encode_du_an đổi cách TRA cứu (theo dự án đã chuẩn hóa); model.predict y nguyên.
# =============================================================================
def normalize_direction(value: str) -> tuple[str, bool]:
    if not value or value == "Không rõ":
        return "Không rõ", False
    text = str(value)
    is_corner = "Căn góc" in text
    text = re.sub(r"Căn góc+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Mã căn\w*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return (text or "Không rõ"), is_corner


def normalize_interior(value: str, is_corner: bool) -> tuple[str, bool]:
    if not value or value == "Không rõ":
        return "Không rõ", False
    return value, is_corner


def set_one_hot(
    vector: dict[str, float],
    features: list[str],
    prefix: str,
    raw_value: str,
    is_corner: bool = False,
    baseline: str | None = None,
) -> None:
    if not raw_value or raw_value == "Không rõ":
        unknown = f"{prefix}_Không rõ"
        if unknown in features:
            vector[unknown] = 1.0
        return

    if baseline and raw_value == baseline:
        return

    if prefix in ("Hướng ban công", "Hướng cửa chính"):
        category, corner_flag = normalize_direction(raw_value)
        if raw_value not in DIRECTION_CHOICES[:-1]:
            corner_flag = is_corner
    elif prefix == "Tình trạng nội thất":
        category, corner_flag = normalize_interior(raw_value, is_corner)
    else:
        category, corner_flag = raw_value, is_corner

    base_col = f"{prefix}_{category}"
    corner_col = f"{prefix}_{category}.1"

    if corner_flag and corner_col in features:
        vector[corner_col] = 1.0
    elif base_col in features:
        vector[base_col] = 1.0
    else:
        unknown = f"{prefix}_Không rõ"
        if unknown in features:
            vector[unknown] = 1.0


def encode_quan(quan: str, encodings: dict) -> float:
    value = encodings["quan_map"].get(quan)
    if value is None:
        return encodings["global_mean"]
    return value


def encode_phuong(quan: str, phuong: str, encodings: dict) -> float:
    value = encodings["phuong_map"].get((quan, phuong))
    if value is not None:
        return value
    return encode_quan(quan, encodings)


def encode_du_an(
    du_an: str,
    quan: str,
    phuong: str,
    encodings: dict,
    use_phuong_fallback: bool,
) -> float:
    """du_an là TÊN DỰ ÁN ĐÃ CHUẨN HÓA. Tra encoding theo (quận,phường,dự án);
    thiếu thì lùi (quận,dự án); cuối cùng lùi về phường (như cũ)."""
    if du_an and du_an != PROJECT_OTHER:
        value = encodings["project_enc_loc"].get((quan, phuong, du_an))
        if value is not None:
            return value
        value = encodings["project_enc_quan"].get((quan, du_an))
        if value is not None:
            return value
    if use_phuong_fallback and phuong:
        return encode_phuong(quan, phuong, encodings)
    return encode_quan(quan, encodings)


def build_input_vector(
    features: list[str],
    encodings: dict,
    *,
    dien_tich_m2: float,
    so_phong_ngu: int,
    so_phong_vs: int,
    tang_so: float | None,
    quan: str,
    phuong: str,
    du_an: str,
    giay_to: str,
    noi_that: str,
    noi_that_can_goc: bool,
    huong_ban_cong: str,
    ban_cong_can_goc: bool,
    huong_cua: str,
    cua_can_goc: bool,
    tinh_trang_bds: str,
) -> dict[str, float]:
    vector = {name: 0.0 for name in features}
    vector["dien_tich_m2"] = float(dien_tich_m2)
    vector["so_phong_ngu"] = float(so_phong_ngu) if so_phong_ngu else np.nan
    vector["so_phong_vs"] = float(so_phong_vs) if so_phong_vs else np.nan
    vector["tang_so"] = float(tang_so) if tang_so is not None else np.nan
    vector["quan_huyen_encoded"] = encode_quan(quan, encodings)
    vector["phuong_xa_encoded"] = encode_phuong(quan, phuong, encodings)
    vector["du_an_encoded"] = encode_du_an(
        du_an,
        quan,
        phuong,
        encodings,
        use_phuong_fallback=True,
    )

    set_one_hot(vector, features, "Giấy tờ pháp lý", giay_to, baseline=LEGAL_BASELINE)
    set_one_hot(
        vector,
        features,
        "Tình trạng nội thất",
        noi_that,
        is_corner=noi_that_can_goc,
    )
    set_one_hot(
        vector,
        features,
        "Hướng ban công",
        huong_ban_cong,
        is_corner=ban_cong_can_goc,
    )
    set_one_hot(
        vector,
        features,
        "Hướng cửa chính",
        huong_cua,
        is_corner=cua_can_goc,
    )
    set_one_hot(
        vector,
        features,
        "Tình trạng bất động sản",
        tinh_trang_bds,
        baseline=STATUS_BASELINE,
    )
    return vector


def format_vnd(value: float) -> str:
    if value >= 1_000_000_000:
        ty = value / 1_000_000_000
        return f"{ty:,.2f} tỷ VND ({value:,.0f} VND)"
    if value >= 1_000_000:
        trieu = value / 1_000_000
        return f"{trieu:,.0f} triệu VND ({value:,.0f} VND)"
    return f"{value:,.0f} VND"


def render_project_selector(quan: str, phuong: str, encodings: dict) -> str:
    """Selector phân cấp: dự án thuộc đúng (quận, phường); nếu phường trống thì
    gom theo quận. Luôn có lựa chọn 'Dự án khác' -> ước lượng theo phường."""
    projects = encodings["project_by_location"].get((quan, phuong), [])
    if not projects:
        projects = encodings["project_by_quan"].get(quan, [])

    options = [PROJECT_OTHER] + projects
    labels = {PROJECT_OTHER: "Dự án khác / không có trong danh sách (ước lượng theo phường)"}

    return st.selectbox(
        "Tên dự án",
        options=options,
        format_func=lambda x: labels.get(x, x),
        help=(
            "Danh sách đã chuẩn hóa và lọc theo Quận/Phường đang chọn. "
            "Chọn 'Dự án khác' nếu căn hộ không thuộc các dự án này — hệ thống "
            "sẽ ước lượng theo giá/m² trung bình của phường."
        ),
    )


# =============================================================================
# NHÀ ĐẤT — feature engineering (target encoding ở log-space, model trả log1p giá)
# =============================================================================
NHADAT_LOAI_HINH = ["Nhà ngõ, hẻm", "Nhà mặt phố, mặt tiền", "Nhà phố liền kề", "Nhà biệt thự"]
NHADAT_PHAP_LY = ["Đã có sổ", "Sổ chung / công chứng vi bằng", "Đang chờ sổ",
                  "Giấy tờ viết tay", "Không có sổ", "Không rõ"]
NHADAT_NOI_THAT = ["Nội thất đầy đủ", "Nội thất cao cấp", "Hoàn thiện cơ bản",
                   "Bàn giao thô", "Không rõ"]
NHADAT_HUONG = ["Đông", "Tây", "Nam", "Bắc", "Đông Bắc", "Tây Bắc",
                "Đông Nam", "Tây Nam", "Không rõ"]
NHADAT_DAC_DIEM = ["Hẻm xe hơi", "Nhà nở hậu", "Nhà tóp hậu", "Nhà nát",
                   "Nhà chưa hoàn công", "Nhà dính quy hoạch / lộ giới",
                   "Đất chưa chuyển thổ", "Hiện trạng khác"]


def nhadat_encode_location(value: str, kind: str, encoders: dict) -> float:
    """Tra target encoding (log-space) cho quận/phường; thiếu thì dùng global."""
    enc = encoders[kind]
    return float(enc["map"].get(value, enc["global"]))


def build_nhadat_vector(
    features: list[str],
    encoders: dict,
    *,
    quan: str,
    phuong: str,
    loai_hinh: str,
    dien_tich_dat: float,
    dien_tich_su_dung: float | None,
    so_phong_ngu: float | None,
    so_phong_vs: float | None,
    tong_so_tang: float | None,
    mat_tien: float | None,
    chieu_dai: float | None,
    duong_rong: float | None,
    co_oto: bool,
    kinh_doanh: bool,
    thang_may: bool,
    lo_goc: bool,
    phap_ly: str,
    noi_that: str,
    huong: str,
    dac_diem: list[str],
) -> dict[str, float]:
    vec = {name: np.nan for name in features}

    # số: thiếu -> NaN (XGBoost tự xử lý)
    vec["dien_tich_dat_m2"] = float(dien_tich_dat)
    vec["dien_tich_m2"] = float(dien_tich_dat)
    vec["dien_tich_su_dung_m2"] = float(dien_tich_su_dung) if dien_tich_su_dung else np.nan
    vec["so_phong_ngu"] = float(so_phong_ngu) if so_phong_ngu else np.nan
    vec["so_phong_vs"] = float(so_phong_vs) if so_phong_vs else np.nan
    vec["tong_so_tang"] = float(tong_so_tang) if tong_so_tang else np.nan
    vec["chieu_dai_m"] = float(chieu_dai) if chieu_dai else np.nan
    # mặt tiền điền cho cả mat_tien_m và chieu_ngang_m (giống lúc train)
    mt = float(mat_tien) if mat_tien else np.nan
    vec["mat_tien_m"] = mt
    vec["chieu_ngang_m"] = mt
    vec["duong_rong_m"] = float(duong_rong) if duong_rong else np.nan

    # cờ nhị phân (không phải one-hot) -> đặt 0/1 tường minh
    vec["co_oto"] = 1.0 if co_oto else 0.0
    vec["kinh_doanh"] = 1.0 if kinh_doanh else 0.0
    vec["thang_may"] = 1.0 if thang_may else 0.0
    vec["lo_goc"] = 1.0 if lo_goc else 0.0

    # one-hot: đặt tất cả nhóm = 0 trước, rồi bật cột được chọn (nếu cột tồn tại)
    for name in features:
        if name.startswith(("loai_hinh_", "Giấy tờ pháp lý_",
                            "Tình trạng nội thất_", "Hướng cửa chính_")) or name.startswith("dd_"):
            vec[name] = 0.0

    def turn_on(col: str) -> None:
        if col in features:
            vec[col] = 1.0

    turn_on(f"loai_hinh_{loai_hinh}")
    if phap_ly != "Không rõ":
        turn_on(f"Giấy tờ pháp lý_{phap_ly}")
    if noi_that != "Không rõ":
        turn_on(f"Tình trạng nội thất_{noi_that}")
    if huong != "Không rõ":
        turn_on(f"Hướng cửa chính_{huong}")
    for dd in dac_diem:
        turn_on(f"dd_{dd}")

    # target encoding vị trí (log-space của giá/m²)
    vec["quan_huyen_te"] = nhadat_encode_location(quan, "quan_huyen", encoders)
    vec["phuong_xa_te"] = nhadat_encode_location(phuong, "phuong_xa", encoders)
    return vec


# =============================================================================
# UI LAYER — Soft UI Evolution + Bento Grid (nền Charcoal)
# =============================================================================
COLOR = {
    "bg": "#1E2226",
    "bg_grad": "#23282D",
    "surface": "#2A3035",
    "surface_2": "#31373D",
    "border": "rgba(255,255,255,0.06)",
    "text": "#E7EAEC",
    "text_muted": "#98A2AB",
    "accent": "#E3B341",
    "accent_soft": "rgba(227,179,65,0.16)",
    "success": "#5BB98C",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
        :root {{
            --surface: {COLOR['surface']}; --surface-2: {COLOR['surface_2']};
            --border: {COLOR['border']}; --text: {COLOR['text']}; --muted: {COLOR['text_muted']};
            --accent: {COLOR['accent']}; --accent-soft: {COLOR['accent_soft']};
            --radius: 20px; --t: 220ms cubic-bezier(.4,0,.2,1);
        }}
        .stApp {{
            background: radial-gradient(1200px 600px at 100% -10%, {COLOR['bg_grad']} 0%, {COLOR['bg']} 55%), {COLOR['bg']};
            color: var(--text); font-family: 'Inter', -apple-system, sans-serif;
        }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
        h1,h2,h3,h4, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {{
            font-family: 'Plus Jakarta Sans', sans-serif !important; color: var(--text) !important;
            letter-spacing: -0.01em;
        }}
        p, span, label, .stMarkdown {{ color: var(--text); }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: linear-gradient(180deg, var(--surface) 0%, #262C31 100%) !important;
            border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
            padding: 1.25rem 1.35rem !important;
            box-shadow: 0 14px 34px rgba(0,0,0,0.42), inset 0 1px 0 rgba(255,255,255,0.04) !important;
            transition: transform var(--t), box-shadow var(--t);
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 20px 44px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06) !important;
        }}
        [data-baseweb="select"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{
            background: var(--surface-2) !important; border: 1px solid var(--border) !important;
            border-radius: 13px !important; color: var(--text) !important;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.30) !important;
            transition: border-color var(--t), box-shadow var(--t);
        }}
        [data-baseweb="select"] > div:hover, [data-testid="stNumberInput"] input:hover {{
            border-color: rgba(227,179,65,0.4) !important;
        }}
        [data-baseweb="select"] svg {{ color: var(--muted) !important; }}
        /* Ẩn nút +/- vuông của number_input cho gọn (vẫn gõ số bình thường) */
        [data-testid="stNumberInput"] button {{ display: none !important; }}
        [data-testid="stNumberInput"] input {{ border-radius: 13px !important; }}
        [data-testid="stWidgetLabel"] p {{ color: var(--muted) !important; font-weight: 500; }}
        .stButton > button {{
            background: linear-gradient(135deg, #EFC04F 0%, #E3B341 100%) !important;
            color: #20262A !important; border: none !important; border-radius: 14px !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 700 !important;
            padding: 0.7rem 1rem !important; cursor: pointer !important;
            box-shadow: 0 8px 20px rgba(227,179,65,0.30) !important;
            transition: transform var(--t), box-shadow var(--t), filter var(--t);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px); filter: brightness(1.03);
            box-shadow: 0 12px 26px rgba(227,179,65,0.42) !important;
        }}
        .stButton > button:active {{ transform: translateY(0); }}
        [data-testid="stMetric"] {{
            background: var(--surface-2); border: 1px solid var(--border); border-radius: 16px;
            padding: 0.85rem 1rem; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }}
        [data-testid="stMetricValue"] {{
            color: var(--accent) !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 800 !important;
        }}
        [data-testid="stMetricLabel"] p {{ color: var(--muted) !important; }}
        [data-testid="stExpander"] {{
            background: var(--surface-2); border: 1px solid var(--border) !important; border-radius: 14px !important;
        }}
        [data-testid="stExpander"] summary {{ color: var(--muted) !important; }}
        [data-baseweb="checkbox"] [data-checked="true"] {{ background: var(--accent) !important; }}
        [data-testid="stAlert"] {{ border-radius: 14px !important; border: 1px solid var(--border) !important; }}
        hr {{ border-color: var(--border) !important; opacity: .6; }}
        @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_header(subtitle_html: str | None = None) -> None:
    default_sub = (
        f'Dự đoán giá bất động sản Hà Nội bằng mô hình '
        f'<b style="color:{COLOR["accent"]}">XGBoost</b>.'
    )
    sub = subtitle_html or default_sub
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLOR['surface']} 0%, #262C31 100%);
            border: 1px solid {COLOR['border']}; border-radius: 24px; padding: 1.6rem 1.8rem;
            margin-bottom: 1.4rem; box-shadow: 0 16px 38px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
            display: flex; align-items: center; gap: 1.1rem;">
            <div style="width:56px;height:56px;flex:0 0 56px;display:flex;align-items:center;
                justify-content:center;background:{COLOR['accent_soft']};border-radius:16px;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{COLOR['accent']}"
                     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="4" y="3" width="16" height="18" rx="1.5"/><path d="M9 21V8h6v13"/>
                    <line x1="8" y1="7" x2="8.01" y2="7"/><line x1="12" y1="7" x2="12.01" y2="7"/>
                    <line x1="16" y1="7" x2="16.01" y2="7"/><line x1="8" y1="11" x2="8.01" y2="11"/>
                    <line x1="16" y1="11" x2="16.01" y2="11"/>
                </svg>
            </div>
            <div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.55rem;font-weight:800;
                    color:{COLOR['text']};line-height:1.1;">Hệ thống định giá Bất động sản Hà Nội</div>
                <div style="color:{COLOR['text_muted']};font-size:0.92rem;margin-top:0.35rem;">{sub}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon_path: str, title: str, subtitle: str = "") -> None:
    sub = (
        f'<div style="color:{COLOR["text_muted"]};font-size:0.8rem;margin-top:1px;">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.75rem;">
            <span style="width:34px;height:34px;display:flex;align-items:center;justify-content:center;
                background:{COLOR['accent_soft']};border-radius:11px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{COLOR['accent']}"
                     stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{icon_path}</svg>
            </span>
            <div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:1.02rem;
                    color:{COLOR['text']};">{title}</div>{sub}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


ICON_AREA = '<path d="M21 3 3 21"/><path d="M21 9V3h-6"/><path d="M3 15v6h6"/>'
ICON_PIN = '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>'
ICON_DOC = '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M9 13h6M9 17h4"/>'
ICON_CHART = '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/>'


def render_chungcu() -> None:
    # ---- Loading: GIỮ NGUYÊN ----
    df = load_training_data()
    model, features = load_model_bundle()
    encodings = build_location_encodings(df)

    col_form, col_result = st.columns([1.15, 0.85], gap="large")

    with col_form:
        # ===== BENTO 1: Diện tích & phòng =====
        with st.container(border=True):
            section_header(ICON_AREA, "Diện tích & phòng", "Thông số cơ bản của căn hộ")
            c1, c2, c3 = st.columns(3)
            with c1:
                dien_tich = st.number_input("Diện tích (m²)", min_value=15.0, max_value=500.0,
                                            value=None, step=1.0, placeholder="VD: 70")
            with c2:
                phong_ngu = st.selectbox("Số phòng ngủ", [1, 2, 3, 4, 5],
                                         index=None, placeholder="Chọn")
            with c3:
                phong_vs = st.selectbox("Số phòng vệ sinh", [1, 2, 3],
                                        index=None, placeholder="Chọn")
            tang_input = st.number_input(
                "Tầng số", min_value=0, max_value=60, value=None, step=1,
                placeholder="Bỏ trống nếu không rõ",
            )
            tang_so = float(tang_input) if tang_input else None

        # ===== BENTO 2: Vị trí (phân cấp Quận -> Phường -> Dự án) =====
        with st.container(border=True):
            section_header(ICON_PIN, "Vị trí", "Chọn lần lượt Quận → Phường → Dự án")
            loc1, loc2 = st.columns(2)
            quan_list = sorted(encodings["quan_map"].keys())
            with loc1:
                quan = st.selectbox("Quận / Huyện", quan_list,
                                    index=None, placeholder="Chọn Quận/Huyện")
            phuong_list = encodings["hierarchy"].get(quan, []) if quan else []
            with loc2:
                phuong = st.selectbox("Phường / Xã", phuong_list, index=None,
                                      placeholder="Chọn Phường/Xã") if phuong_list else None

            du_an = render_project_selector(quan, phuong, encodings) if (quan and phuong) else PROJECT_OTHER

        # ===== BENTO 3: Đặc điểm bất động sản =====
        with st.container(border=True):
            section_header(ICON_DOC, "Đặc điểm bất động sản", "Pháp lý, nội thất, hướng & tình trạng")
            d1, d2 = st.columns(2)
            with d1:
                giay_to = st.selectbox("Giấy tờ pháp lý", LEGAL_CHOICES,
                                       index=None, placeholder="Chọn")
            with d2:
                tinh_trang_bds = st.selectbox("Tình trạng bất động sản", STATUS_CHOICES,
                                              index=None, placeholder="Chọn")
            noi_that = st.selectbox("Tình trạng nội thất", INTERIOR_CHOICES,
                                    index=None, placeholder="Chọn")
            noi_that_can_goc = st.checkbox("Căn góc (nội thất)")
            hc1, hc2 = st.columns(2)
            with hc1:
                huong_ban_cong = st.selectbox("Hướng ban công", DIRECTION_CHOICES,
                                              index=None, placeholder="Chọn")
                ban_cong_can_goc = st.checkbox("Căn góc (ban công)")
            with hc2:
                huong_cua = st.selectbox("Hướng cửa chính", DIRECTION_CHOICES,
                                         index=None, placeholder="Chọn")
                cua_can_goc = st.checkbox("Căn góc (cửa chính)")
            predict_clicked = st.button("Định giá ngay", type="primary", use_container_width=True)

    with col_result:
        # ===== BENTO: Kết quả =====
        with st.container(border=True):
            section_header(ICON_CHART, "Kết quả định giá", "Ước lượng từ mô hình")
            if predict_clicked and (not quan or not phuong or not dien_tich):
                st.warning("Vui lòng nhập **Diện tích**, **Quận/Huyện** và **Phường/Xã** trước khi định giá.")
            elif predict_clicked:
                scroll_to_top()
                vector = build_input_vector(
                    features, encodings,
                    dien_tich_m2=dien_tich,
                    so_phong_ngu=phong_ngu, so_phong_vs=phong_vs,
                    tang_so=tang_so, quan=quan, phuong=phuong, du_an=du_an,
                    giay_to=giay_to or "Không rõ", noi_that=noi_that or "Không rõ",
                    noi_that_can_goc=noi_that_can_goc,
                    huong_ban_cong=huong_ban_cong or "Không rõ", ban_cong_can_goc=ban_cong_can_goc,
                    huong_cua=huong_cua or "Không rõ", cua_can_goc=cua_can_goc,
                    tinh_trang_bds=tinh_trang_bds or "Không rõ",
                )
                input_df = pd.DataFrame([vector])[features]
                predicted_price = float(model.predict(input_df)[0])
                predicted_m2 = predicted_price / dien_tich if dien_tich else 0.0
                band = 0.15  # khoảng tham khảo ±15%

                st.success("Dự đoán hoàn tất")
                st.metric("Giá bán dự kiến", format_vnd(predicted_price))
                st.caption(
                    f"Khoảng tham khảo: **{format_vnd(predicted_price*(1-band))}** "
                    f"– **{format_vnd(predicted_price*(1+band))}**"
                )
                st.metric("Giá/m² dự kiến", f"{predicted_m2:,.0f} VND/m²")
            else:
                st.markdown(
                    "Nhập thông tin căn hộ bên trái, chọn **Quận → Phường → Dự án** "
                    "rồi bấm **Định giá ngay** để xem mức giá ước lượng."
                )

        # ===== BENTO: Mẹo định giá =====
        with st.container(border=True):
            section_header(ICON_CHART, "Điều gì ảnh hưởng tới giá?", "Một vài yếu tố đáng lưu ý")
            st.markdown(
                "- **Vị trí** (quận, phường, dự án) là yếu tố chi phối lớn nhất.\n"
                "- **Pháp lý rõ ràng** (sổ hồng riêng) thường được định giá cao hơn.\n"
                "- **Tầng, hướng và nội thất** tạo chênh lệch giữa các căn cùng toà.\n"
                "- Giá là **ước lượng tham khảo** — nên đối chiếu thêm tin rao thực tế."
            )


def render_nhadat() -> None:
    model, features, encoders = load_nhadat_bundle()
    meta = load_nhadat_hierarchy()
    hierarchy = meta["hierarchy"]

    col_form, col_result = st.columns([1.15, 0.85], gap="large")

    with col_form:
        # ===== BENTO 1: Loại hình & diện tích =====
        with st.container(border=True):
            section_header(ICON_AREA, "Loại hình & diện tích", "Thông tin cơ bản của nhà/đất")
            loai_hinh = st.selectbox("Loại hình", NHADAT_LOAI_HINH,
                                     index=None, placeholder="Chọn loại hình")
            a1, a2 = st.columns(2)
            with a1:
                dien_tich_dat = st.number_input("Diện tích đất (m²)", min_value=5.0,
                                                max_value=2000.0, value=None, step=1.0,
                                                placeholder="VD: 45")
            with a2:
                dt_sd_in = st.number_input("Diện tích sử dụng (m²)",
                                           min_value=0.0, max_value=5000.0, value=None,
                                           step=1.0, placeholder="Bỏ trống nếu không rõ")
            b1, b2 = st.columns(2)
            with b1:
                mt_in = st.number_input("Mặt tiền (m)", min_value=0.0, max_value=100.0,
                                        value=None, step=0.1, placeholder="VD: 4")
            with b2:
                cd_in = st.number_input("Chiều dài (m)", min_value=0.0, max_value=200.0,
                                        value=None, step=0.1, placeholder="Bỏ trống nếu không rõ")
            dr_in = st.number_input("Độ rộng đường/ngõ trước nhà (m)",
                                    min_value=0.0, max_value=100.0, value=None, step=0.1,
                                    placeholder="Bỏ trống nếu không rõ")
            c1, c2, c3 = st.columns(3)
            with c1:
                pn_in = st.number_input("Số phòng ngủ", min_value=0, max_value=20,
                                        value=None, step=1, placeholder="—")
            with c2:
                vs_in = st.number_input("Số phòng VS", min_value=0, max_value=20,
                                        value=None, step=1, placeholder="—")
            with c3:
                tang_in = st.number_input("Tổng số tầng", min_value=0, max_value=50,
                                          value=None, step=1, placeholder="—")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                co_oto = st.checkbox("Ô tô vào/đỗ")
            with f2:
                kinh_doanh = st.checkbox("Tiện kinh doanh")
            with f3:
                thang_may = st.checkbox("Có thang máy")
            with f4:
                lo_goc = st.checkbox("Lô góc")

        # ===== BENTO 2: Vị trí (phân cấp Quận -> Phường) =====
        with st.container(border=True):
            section_header(ICON_PIN, "Vị trí", "Chọn lần lượt Quận → Phường")
            quan_list = sorted(hierarchy.keys())
            loc1, loc2 = st.columns(2)
            with loc1:
                quan = st.selectbox("Quận / Huyện", quan_list, index=None,
                                    placeholder="Chọn Quận/Huyện", key="nd_quan")
            phuong_list = hierarchy.get(quan, []) if quan else []
            with loc2:
                phuong = st.selectbox("Phường / Xã", phuong_list, index=None,
                                      placeholder="Chọn Phường/Xã", key="nd_phuong") if phuong_list else None

        # ===== BENTO 3: Pháp lý, nội thất, đặc điểm =====
        with st.container(border=True):
            section_header(ICON_DOC, "Pháp lý & đặc điểm", "Giấy tờ, nội thất, hướng, đặc điểm nhà")
            d1, d2 = st.columns(2)
            with d1:
                phap_ly = st.selectbox("Giấy tờ pháp lý", NHADAT_PHAP_LY,
                                       index=None, placeholder="Chọn")
            with d2:
                noi_that = st.selectbox("Tình trạng nội thất", NHADAT_NOI_THAT,
                                        index=None, placeholder="Chọn")
            huong = st.selectbox("Hướng cửa chính", NHADAT_HUONG,
                                 index=None, placeholder="Chọn")
            dac_diem = st.multiselect("Đặc điểm (chọn nhiều)", NHADAT_DAC_DIEM, default=[])
            predict_clicked = st.button("Định giá ngay", type="primary",
                                        use_container_width=True, key="nd_btn")

    with col_result:
        with st.container(border=True):
            section_header(ICON_CHART, "Kết quả định giá", "Ước lượng từ mô hình nhà đất")
            if predict_clicked and (not loai_hinh or not dien_tich_dat or not quan or not phuong):
                st.warning("Vui lòng nhập **Loại hình**, **Diện tích đất**, **Quận/Huyện** và **Phường/Xã**.")
            elif predict_clicked:
                scroll_to_top()
                vec = build_nhadat_vector(
                    features, encoders,
                    quan=quan, phuong=phuong, loai_hinh=loai_hinh,
                    dien_tich_dat=dien_tich_dat,
                    dien_tich_su_dung=dt_sd_in or None,
                    so_phong_ngu=pn_in or None, so_phong_vs=vs_in or None,
                    tong_so_tang=tang_in or None,
                    mat_tien=mt_in or None, chieu_dai=cd_in or None,
                    duong_rong=dr_in or None,
                    co_oto=co_oto, kinh_doanh=kinh_doanh,
                    thang_may=thang_may, lo_goc=lo_goc,
                    phap_ly=phap_ly or "Không rõ", noi_that=noi_that or "Không rõ",
                    huong=huong or "Không rõ", dac_diem=dac_diem,
                )
                input_df = pd.DataFrame([vec])[features]
                # model trả log1p(giá/m²) -> giá/m² rồi nhân diện tích ra tổng giá
                pred_ppm2 = float(np.expm1(model.predict(input_df)[0]))
                predicted_price = pred_ppm2 * dien_tich_dat
                predicted_m2 = pred_ppm2
                band = 0.22  # khoảng tham khảo ±22% (theo MAPE)

                st.success("Dự đoán hoàn tất")
                st.metric("Giá bán dự kiến", format_vnd(predicted_price))
                st.caption(
                    f"Khoảng tham khảo: **{format_vnd(predicted_price*(1-band))}** "
                    f"– **{format_vnd(predicted_price*(1+band))}**"
                )
                st.metric("Giá/m² đất dự kiến", f"{predicted_m2:,.0f} VND/m²")
            else:
                st.markdown(
                    "Nhập thông tin nhà/đất bên trái, chọn **Loại hình → Diện tích → "
                    "Quận → Phường** rồi bấm **Định giá ngay**."
                )

        # ===== BENTO: Mẹo định giá =====
        with st.container(border=True):
            section_header(ICON_CHART, "Điều gì ảnh hưởng tới giá?", "Một vài yếu tố đáng lưu ý")
            st.markdown(
                "- **Pháp lý** tác động mạnh nhất: nhà *không sổ* bị giảm giá rõ rệt.\n"
                "- **Ô tô vào nhà** và **vị trí kinh doanh** làm tăng giá đáng kể.\n"
                "- **Mặt tiền & độ rộng đường/ngõ** quyết định giá nhà phố/ngõ.\n"
                "- Điền càng đầy đủ, ước lượng càng sát; bỏ trống vẫn dự đoán được."
            )


def main() -> None:
    st.set_page_config(page_title="Định giá Bất động sản Hà Nội", page_icon="🏢", layout="wide")
    inject_css()

    mode = st.radio(
        "Chọn loại bất động sản",
        ["🏢 Chung cư", "🏠 Nhà đất"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if "Chung cư" in mode:
        hero_header(
            f'Định giá <b style="color:{COLOR["accent"]}">căn hộ chung cư</b> — '
            "chọn vị trí phân cấp Quận / Phường / Dự án."
        )
        render_chungcu()
    else:
        hero_header(
            f'Định giá <b style="color:{COLOR["accent"]}">nhà đất thổ cư</b> — '
            "nhà ngõ/hẻm, mặt phố, liền kề, biệt thự theo Quận / Phường."
        )
        render_nhadat()


if __name__ == "__main__":
    main()
