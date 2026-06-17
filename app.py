from __future__ import annotations

import pickle
import re
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

    du_an_map = (
        df.dropna(subset=["du_an"])
        .groupby("du_an")["du_an_encoded"]
        .first()
        .astype(float)
        .to_dict()
    )

    hierarchy: dict[str, list[str]] = {}
    for quan in sorted(quan_map.keys()):
        phuongs = sorted(
            phuong_subset.loc[phuong_subset["quan_huyen"] == quan, "phuong_xa"].unique()
        )
        hierarchy[quan] = phuongs

    du_an_by_location: dict[tuple[str, str], list[str]] = {}
    for _, row in df[["quan_huyen", "phuong_xa", "du_an"]].drop_duplicates().iterrows():
        if pd.isna(row["quan_huyen"]) or pd.isna(row["du_an"]):
            continue
        phuong = row["phuong_xa"] if pd.notna(row["phuong_xa"]) else ""
        key = (row["quan_huyen"], phuong)
        du_an_by_location.setdefault(key, [])
        title = str(row["du_an"])
        if title not in du_an_by_location[key]:
            du_an_by_location[key].append(title)

    global_mean = float(df["gia_tren_m2"].mean())
    return {
        "quan_map": quan_map,
        "phuong_map": phuong_map,
        "du_an_map": du_an_map,
        "hierarchy": hierarchy,
        "du_an_by_location": du_an_by_location,
        "global_mean": global_mean,
    }


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
    if du_an and du_an != "__new__":
        mapped = encodings["du_an_map"].get(du_an)
        if mapped is not None:
            return mapped
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


def shorten_title(title: str, max_len: int = 70) -> str:
    clean = re.sub(r"\s+", " ", str(title)).strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1] + "…"


def render_project_selector(
    quan: str,
    phuong: str,
    encodings: dict,
) -> str:
    key = (quan, phuong)
    projects = encodings["du_an_by_location"].get(key, [])
    if not projects:
        projects = [
            title
            for (q, _), titles in encodings["du_an_by_location"].items()
            if q == quan
            for title in titles
        ]

    options = ["__new__"] + projects
    labels = {
        "__new__": "Dự án mới / không có trong dữ liệu (ước lượng theo phường)",
    }
    for project in projects:
        labels[project] = shorten_title(project)

    return st.selectbox(
        "Dự án / tin đăng tham chiếu",
        options=options,
        format_func=lambda x: labels.get(x, shorten_title(x)),
        help=(
            "Chọn dự án có sẵn trong dữ liệu để dùng mã hóa `du_an_encoded`. "
            "Nếu chọn dự án mới, hệ thống ước lượng theo giá/m² trung bình của phường."
        ),
    )


def main() -> None:
    st.set_page_config(
        page_title="Định giá chung cư Hà Nội",
        page_icon="🏢",
        layout="wide",
    )

    st.title("🏢 Định giá chung cư Hà Nội")
    st.caption(
        "Dự đoán giá bán căn hộ dựa trên mô hình XGBoost và bảng mã hóa "
        "Quận / Phường / Dự án từ file `chungcu_ready_to_train.xlsx`."
    )

    df = load_training_data()
    model, features = load_model_bundle()
    encodings = build_location_encodings(df)

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.subheader("Thông tin căn hộ")

        c1, c2, c3 = st.columns(3)
        with c1:
            dien_tich = st.number_input(
                "Diện tích (m²)",
                min_value=15.0,
                max_value=500.0,
                value=70.0,
                step=1.0,
            )
        with c2:
            phong_ngu = st.selectbox("Số phòng ngủ", [1, 2, 3, 4, 5], index=1)
        with c3:
            phong_vs = st.selectbox("Số phòng vệ sinh", [1, 2, 3], index=1)

        tang_input = st.number_input(
            "Tầng số (để trống nếu không biết)",
            min_value=0,
            max_value=60,
            value=0,
            step=1,
            help="Nhập 0 nếu không biết tầng — mô hình sẽ xử lý như dữ liệu thiếu.",
        )
        tang_so = None if tang_input == 0 else float(tang_input)

        st.markdown("---")
        st.markdown("**Vị trí**")

        quan_list = sorted(encodings["quan_map"].keys())
        quan = st.selectbox("Quận / Huyện", quan_list)

        phuong_list = encodings["hierarchy"].get(quan, [])
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
            st.write(f"**Dự án** → `{d_enc:,.0f}` VND/m²")

        st.markdown("---")
        st.markdown("**Đặc điểm bất động sản**")

        giay_to = st.selectbox("Giấy tờ pháp lý", LEGAL_CHOICES, index=0)
        noi_that = st.selectbox("Tình trạng nội thất", INTERIOR_CHOICES, index=0)
        noi_that_can_goc = st.checkbox("Căn góc (nội thất)")

        hc1, hc2 = st.columns(2)
        with hc1:
            huong_ban_cong = st.selectbox("Hướng ban công", DIRECTION_CHOICES, index=7)
            ban_cong_can_goc = st.checkbox("Căn góc (ban công)")
        with hc2:
            huong_cua = st.selectbox("Hướng cửa chính", DIRECTION_CHOICES, index=7)
            cua_can_goc = st.checkbox("Căn góc (cửa chính)")

        tinh_trang_bds = st.selectbox("Tình trạng bất động sản", STATUS_CHOICES, index=0)

        predict_clicked = st.button("🔮 Dự đoán giá", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Kết quả định giá")

        if predict_clicked:
            vector = build_input_vector(
                features,
                encodings,
                dien_tich_m2=dien_tich,
                so_phong_ngu=phong_ngu,
                so_phong_vs=phong_vs,
                tang_so=tang_so,
                quan=quan,
                phuong=phuong,
                du_an=du_an,
                giay_to=giay_to,
                noi_that=noi_that,
                noi_that_can_goc=noi_that_can_goc,
                huong_ban_cong=huong_ban_cong,
                ban_cong_can_goc=ban_cong_can_goc,
                huong_cua=huong_cua,
                cua_can_goc=cua_can_goc,
                tinh_trang_bds=tinh_trang_bds,
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
                Nhập thông tin căn hộ bên trái và bấm **Dự đoán giá**.

                **Cách mã hóa vị trí**
                - `quan_huyen_encoded`: giá/m² trung bình theo quận/huyện
                - `phuong_xa_encoded`: giá/m² trung bình theo phường/xã
                - `du_an_encoded`: giá/m² của dự án đã chọn; nếu là dự án mới thì dùng giá/m² trung bình phường
                """
            )

        st.markdown("---")
        st.markdown("**Thống kê dữ liệu huấn luyện**")
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Số tin đăng", f"{len(df):,}")
        stat2.metric("Số quận/huyện", f"{len(encodings['quan_map']):,}")
        stat3.metric("Số dự án", f"{len(encodings['du_an_map']):,}")


if __name__ == "__main__":
    main()
