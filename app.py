from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "chungcu_ready_to_train.xlsx"
MODEL_PATH = BASE_DIR / "model_chung_cu.pkl"
FEATURES_PATH = BASE_DIR / "features_chung_cu.pkl"

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
    vector["so_phong_ngu"] = float(so_phong_ngu)
    vector["so_phong_vs"] = float(so_phong_vs)
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
        [data-testid="stNumberInput"] button {{
            background: var(--surface-2) !important; border: 1px solid var(--border) !important;
        }}
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


def hero_header() -> None:
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
                    color:{COLOR['text']};line-height:1.1;">Định giá chung cư Hà Nội</div>
                <div style="color:{COLOR['text_muted']};font-size:0.92rem;margin-top:0.35rem;">
                    Dự đoán giá bán căn hộ bằng mô hình <b style="color:{COLOR['accent']}">XGBoost</b> &
                    chọn vị trí phân cấp Quận / Phường / Dự án.</div>
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


def main() -> None:
    st.set_page_config(page_title="Định giá chung cư Hà Nội", page_icon="🏢", layout="wide")
    inject_css()

    # ---- Loading: GIỮ NGUYÊN ----
    df = load_training_data()
    model, features = load_model_bundle()
    encodings = build_location_encodings(df)

    hero_header()
    col_form, col_result = st.columns([1.15, 0.85], gap="large")

    with col_form:
        # ===== BENTO 1: Diện tích & phòng =====
        with st.container(border=True):
            section_header(ICON_AREA, "Diện tích & phòng", "Thông số cơ bản của căn hộ")
            c1, c2, c3 = st.columns(3)
            with c1:
                dien_tich = st.number_input("Diện tích (m²)", min_value=15.0, max_value=500.0,
                                            value=70.0, step=1.0)
            with c2:
                phong_ngu = st.selectbox("Số phòng ngủ", [1, 2, 3, 4, 5], index=1)
            with c3:
                phong_vs = st.selectbox("Số phòng vệ sinh", [1, 2, 3], index=1)
            tang_input = st.number_input(
                "Tầng số (để trống nếu không biết)", min_value=0, max_value=60, value=0, step=1,
                help="Nhập 0 nếu không biết tầng — mô hình sẽ xử lý như dữ liệu thiếu.",
            )
            tang_so = None if tang_input == 0 else float(tang_input)

        # ===== BENTO 2: Vị trí (phân cấp Quận -> Phường -> Dự án) =====
        with st.container(border=True):
            section_header(ICON_PIN, "Vị trí", "Chọn lần lượt Quận → Phường → Dự án")
            loc1, loc2 = st.columns(2)
            quan_list = sorted(encodings["quan_map"].keys())
            with loc1:
                quan = st.selectbox("Quận / Huyện", quan_list)
            phuong_list = encodings["hierarchy"].get(quan, [])
            with loc2:
                if not phuong_list:
                    st.warning("Không có dữ liệu phường cho quận này.")
                    phuong = ""
                else:
                    phuong = st.selectbox("Phường / Xã", phuong_list)

            du_an = render_project_selector(quan, phuong, encodings)

            q_enc = encode_quan(quan, encodings)
            p_enc = encode_phuong(quan, phuong, encodings) if phuong else q_enc
            d_enc = encode_du_an(du_an, quan, phuong, encodings, use_phuong_fallback=True)
            with st.expander("Xem giá trị mã hóa vị trí (target encoding giá/m²)"):
                st.write(f"**Quận** → `{q_enc:,.0f}` VND/m²")
                if phuong:
                    st.write(f"**Phường** → `{p_enc:,.0f}` VND/m²")
                proj_label = "Dự án khác (theo phường)" if du_an == PROJECT_OTHER else du_an
                st.write(f"**{proj_label}** → `{d_enc:,.0f}` VND/m²")

        # ===== BENTO 3: Đặc điểm bất động sản =====
        with st.container(border=True):
            section_header(ICON_DOC, "Đặc điểm bất động sản", "Pháp lý, nội thất, hướng & tình trạng")
            d1, d2 = st.columns(2)
            with d1:
                giay_to = st.selectbox("Giấy tờ pháp lý", LEGAL_CHOICES, index=0)
            with d2:
                tinh_trang_bds = st.selectbox("Tình trạng bất động sản", STATUS_CHOICES, index=0)
            noi_that = st.selectbox("Tình trạng nội thất", INTERIOR_CHOICES, index=0)
            noi_that_can_goc = st.checkbox("Căn góc (nội thất)")
            hc1, hc2 = st.columns(2)
            with hc1:
                huong_ban_cong = st.selectbox("Hướng ban công", DIRECTION_CHOICES, index=7)
                ban_cong_can_goc = st.checkbox("Căn góc (ban công)")
            with hc2:
                huong_cua = st.selectbox("Hướng cửa chính", DIRECTION_CHOICES, index=7)
                cua_can_goc = st.checkbox("Căn góc (cửa chính)")
            predict_clicked = st.button("Định giá ngay", type="primary", use_container_width=True)

    with col_result:
        # ===== BENTO: Kết quả =====
        with st.container(border=True):
            section_header(ICON_CHART, "Kết quả định giá", "Ước lượng từ mô hình")
            if predict_clicked:
                vector = build_input_vector(
                    features, encodings,
                    dien_tich_m2=dien_tich, so_phong_ngu=phong_ngu, so_phong_vs=phong_vs,
                    tang_so=tang_so, quan=quan, phuong=phuong, du_an=du_an,
                    giay_to=giay_to, noi_that=noi_that, noi_that_can_goc=noi_that_can_goc,
                    huong_ban_cong=huong_ban_cong, ban_cong_can_goc=ban_cong_can_goc,
                    huong_cua=huong_cua, cua_can_goc=cua_can_goc, tinh_trang_bds=tinh_trang_bds,
                )
                input_df = pd.DataFrame([vector])[features]
                predicted_price = float(model.predict(input_df)[0])
                predicted_m2 = predicted_price / dien_tich if dien_tich else 0.0

                st.success("Dự đoán hoàn tất")
                st.metric("Giá bán dự kiến", format_vnd(predicted_price))
                st.metric("Giá/m² dự kiến", f"{predicted_m2:,.0f} VND/m²")
                st.info(
                    "Giá trị mang tính tham khảo, dựa trên dữ liệu lịch sử và mô hình "
                    "hồi quy XGBoost. Vui lòng kết hợp thêm khảo sát thực tế."
                )
            else:
                st.markdown(
                    """
                    Chọn **Quận → Phường → Dự án** rồi bấm **Định giá ngay**.

                    **Cách mã hóa vị trí**
                    - `quan_huyen_encoded`: giá/m² trung bình theo quận/huyện
                    - `phuong_xa_encoded`: giá/m² trung bình theo phường/xã
                    - `du_an_encoded`: giá/m² của dự án đã chuẩn hóa; nếu chọn *Dự án khác* thì dùng giá/m² trung bình phường
                    """
                )

        # ===== BENTO: Thống kê dữ liệu =====
        with st.container(border=True):
            section_header(ICON_DOC, "Dữ liệu huấn luyện", "Quy mô tập dữ liệu")
            stat1, stat2, stat3 = st.columns(3)
            stat1.metric("Số tin đăng", f"{len(df):,}")
            stat2.metric("Số quận/huyện", f"{len(encodings['quan_map']):,}")
            stat3.metric("Dự án chuẩn hóa", f"{encodings['n_projects']:,}")


if __name__ == "__main__":
    main()
