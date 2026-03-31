
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Conciliação REDE x AUTCOM", layout="wide")


# =========================
# CONFIG / HELPERS
# =========================
REDE_SHEET = "REDE"
AUTCOM_SHEET = "AUTCOM"

COL_REDE_DATA = "data do recebimento"
COL_REDE_NSU = "NSU/CV"
COL_REDE_AUT = "número da autorização"
COL_REDE_BRU = "valor bruto da parcela original"
COL_REDE_LIQ = "valor líquido da parcela"

COL_AUT_DATA = "DTA.VEN"
COL_AUT_NSU = "NÚM.NSU"
COL_AUT_AUT = "LIBERAÇÃO"
COL_AUT_BRU = "VLR.BRU"
COL_AUT_LIQ = "VLR.LÍQ"


def brl(v) -> str:
    if pd.isna(v):
        return ""
    s = f"{float(v):,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def format_date(v) -> str:
    if pd.isna(v):
        return ""
    try:
        dt = pd.to_datetime(v)
        base = dt.strftime("%d/%m/%Y")
        wd = dt.weekday()
        if wd == 5:
            return f"{base} (sáb)"
        if wd == 6:
            return f"{base} (dom)"
        return base
    except Exception:
        return str(v)


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s in {"-", "nan", "None", ""}:
        return ""
    # elimina decimais desnecessários do Excel, ex.: 12345.0
    try:
        f = float(str(s).replace(",", "."))
        if f.is_integer():
            s = str(int(f))
    except Exception:
        pass
    return "".join(ch for ch in s if ch.isdigit())


def normalize_number(value) -> Optional[float]:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        s = value.strip()
        if s in {"", "-", "nan", "None"}:
            return np.nan
        s = s.replace("R$", "").replace(".", "").replace(",", ".").strip()
        try:
            return round(float(s), 2)
        except Exception:
            return np.nan
    try:
        return round(float(value), 2)
    except Exception:
        return np.nan


def normalize_date(value):
    if pd.isna(value):
        return pd.NaT
    try:
        return pd.to_datetime(value).normalize()
    except Exception:
        return pd.NaT


def safe_columns(df: pd.DataFrame, required: List[str], sheet_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"A aba '{sheet_name}' não contém as colunas obrigatórias: {', '.join(missing)}"
        )


def load_excel(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    if REDE_SHEET not in xls.sheet_names or AUTCOM_SHEET not in xls.sheet_names:
        raise ValueError("O arquivo precisa conter as abas 'REDE' e 'AUTCOM'.")

    rede = pd.read_excel(io.BytesIO(file_bytes), sheet_name=REDE_SHEET)
    aut = pd.read_excel(io.BytesIO(file_bytes), sheet_name=AUTCOM_SHEET)

    safe_columns(
        rede,
        [COL_REDE_DATA, COL_REDE_NSU, COL_REDE_AUT, COL_REDE_BRU, COL_REDE_LIQ],
        REDE_SHEET,
    )
    safe_columns(
        aut,
        [COL_AUT_DATA, COL_AUT_NSU, COL_AUT_AUT, COL_AUT_BRU, COL_AUT_LIQ],
        AUTCOM_SHEET,
    )

    # normalização REDE
    rede = rede.copy()
    rede["_idx_rede"] = np.arange(len(rede))
    rede["_data_rede"] = rede[COL_REDE_DATA].apply(normalize_date)
    rede["_nsu"] = rede[COL_REDE_NSU].apply(normalize_id)
    rede["_autorizacao"] = rede[COL_REDE_AUT].apply(normalize_id)
    rede["_vlr_bru"] = rede[COL_REDE_BRU].apply(normalize_number)
    rede["_vlr_liq"] = rede[COL_REDE_LIQ].apply(normalize_number)

    # normalização AUTCOM
    aut = aut.copy()
    aut["_idx_aut"] = np.arange(len(aut))
    aut["_data_aut"] = aut[COL_AUT_DATA].apply(normalize_date)
    aut["_nsu"] = aut[COL_AUT_NSU].apply(normalize_id)
    aut["_autorizacao"] = aut[COL_AUT_AUT].apply(normalize_id)
    aut["_vlr_bru"] = aut[COL_AUT_BRU].apply(normalize_number)
    aut["_vlr_liq"] = aut[COL_AUT_LIQ].apply(normalize_number)

    return rede, aut


def build_indexes(aut: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
    indexes = {
        "nsu": {},
        "aut": {},
        "bru": {},
        "liq": {},
        "bru_liq": {},
    }

    for i, row in aut.iterrows():
        nsu = row["_nsu"]
        autz = row["_autorizacao"]
        bru = row["_vlr_bru"]
        liq = row["_vlr_liq"]

        if nsu:
            indexes["nsu"].setdefault(nsu, []).append(i)
        if autz:
            indexes["aut"].setdefault(autz, []).append(i)
        if not pd.isna(bru):
            indexes["bru"].setdefault(f"{bru:.2f}", []).append(i)
        if not pd.isna(liq):
            indexes["liq"].setdefault(f"{liq:.2f}", []).append(i)
        if not pd.isna(bru) and not pd.isna(liq):
            indexes["bru_liq"].setdefault(f"{bru:.2f}|{liq:.2f}", []).append(i)

    return indexes


def candidate_indices_for_row(row: pd.Series, idx: Dict[str, Dict[str, List[int]]]) -> set:
    candidates = set()

    nsu = row["_nsu"]
    autz = row["_autorizacao"]
    bru = row["_vlr_bru"]
    liq = row["_vlr_liq"]

    if nsu:
        candidates.update(idx["nsu"].get(nsu, []))
    if autz:
        candidates.update(idx["aut"].get(autz, []))
    if not pd.isna(bru):
        candidates.update(idx["bru"].get(f"{bru:.2f}", []))
    if not pd.isna(liq):
        candidates.update(idx["liq"].get(f"{liq:.2f}", []))
    if not pd.isna(bru) and not pd.isna(liq):
        candidates.update(idx["bru_liq"].get(f"{bru:.2f}|{liq:.2f}", []))

    return candidates


def score_pair(rede_row: pd.Series, aut_row: pd.Series) -> Tuple[float, Dict[str, bool]]:
    nsu_ok = bool(rede_row["_nsu"]) and rede_row["_nsu"] == aut_row["_nsu"]
    aut_ok = bool(rede_row["_autorizacao"]) and rede_row["_autorizacao"] == aut_row["_autorizacao"]

    bru_ok = (
        not pd.isna(rede_row["_vlr_bru"])
        and not pd.isna(aut_row["_vlr_bru"])
        and abs(rede_row["_vlr_bru"] - aut_row["_vlr_bru"]) <= 0.01
    )
    liq_ok = (
        not pd.isna(rede_row["_vlr_liq"])
        and not pd.isna(aut_row["_vlr_liq"])
        and abs(rede_row["_vlr_liq"] - aut_row["_vlr_liq"]) <= 0.01
    )

    score = 0.0
    if nsu_ok:
        score += 100
    if aut_ok:
        score += 80
    if bru_ok:
        score += 25
    if liq_ok:
        score += 25

    data_rede = rede_row["_data_rede"]
    data_aut = aut_row["_data_aut"]
    if not pd.isna(data_rede) and not pd.isna(data_aut):
        diff_days = abs((data_rede - data_aut).days)
        if diff_days == 0:
            score += 8
        elif diff_days <= 1:
            score += 5
        elif diff_days <= 3:
            score += 2

    # penalidade se só houver 1 valor batendo sem identificador
    if not nsu_ok and not aut_ok and (bru_ok ^ liq_ok):
        score -= 15

    details = {
        "nsu_ok": nsu_ok,
        "aut_ok": aut_ok,
        "bru_ok": bru_ok,
        "liq_ok": liq_ok,
    }
    return score, details


def robust_match(rede: pd.DataFrame, aut: pd.DataFrame) -> pd.DataFrame:
    idx = build_indexes(aut)
    used_aut = set()
    matches = []

    # prioriza linhas mais "ricas" em identificação
    rede_work = rede.copy()
    rede_work["_priority"] = (
        rede_work["_nsu"].astype(str).str.len().gt(0).astype(int) * 4
        + rede_work["_autorizacao"].astype(str).str.len().gt(0).astype(int) * 3
        + rede_work["_vlr_bru"].notna().astype(int) * 2
        + rede_work["_vlr_liq"].notna().astype(int) * 2
    )
    rede_work = rede_work.sort_values(["_priority", "_data_rede"], ascending=[False, True])

    for rede_i, rede_row in rede_work.iterrows():
        candidates = candidate_indices_for_row(rede_row, idx)
        best_idx = None
        best_score = -999
        best_details = {"nsu_ok": False, "aut_ok": False, "bru_ok": False, "liq_ok": False}

        for aut_i in candidates:
            if aut_i in used_aut:
                continue
            aut_row = aut.loc[aut_i]
            score, details = score_pair(rede_row, aut_row)
            if score > best_score:
                best_score = score
                best_idx = aut_i
                best_details = details

        # regra mínima de aceite:
        # 1 identificador OU os 2 valores ao mesmo tempo
        accepted = (
            best_idx is not None
            and (
                best_details["nsu_ok"]
                or best_details["aut_ok"]
                or (best_details["bru_ok"] and best_details["liq_ok"])
            )
        )

        if accepted:
            used_aut.add(best_idx)
            aut_row = aut.loc[best_idx]

            data_rede = rede_row["_data_rede"]
            data_aut = aut_row["_data_aut"]
            diff_data = np.nan
            if not pd.isna(data_rede) and not pd.isna(data_aut):
                diff_data = int((data_rede - data_aut).days)

            diff_bru = np.nan
            if not pd.isna(rede_row["_vlr_bru"]) and not pd.isna(aut_row["_vlr_bru"]):
                diff_bru = round(rede_row["_vlr_bru"] - aut_row["_vlr_bru"], 2)

            diff_liq = np.nan
            if not pd.isna(rede_row["_vlr_liq"]) and not pd.isna(aut_row["_vlr_liq"]):
                diff_liq = round(rede_row["_vlr_liq"] - aut_row["_vlr_liq"], 2)

            matches.append(
                {
                    "_idx_rede": rede_row["_idx_rede"],
                    "_idx_aut": aut_row["_idx_aut"],
                    "Status geral": "✅ Encontrado",
                    "NSU encontrado": "✅" if best_details["nsu_ok"] else "❌",
                    "Autorização encontrada": "✅" if best_details["aut_ok"] else "❌",
                    "Valor bruto encontrado": "✅" if best_details["bru_ok"] else "❌",
                    "Valor líquido encontrado": "✅" if best_details["liq_ok"] else "❌",
                    "Data REDE": rede_row[COL_REDE_DATA],
                    "Data AUTCOM": aut_row[COL_AUT_DATA],
                    "Diferença de datas (dias)": diff_data,
                    "NSU/CV REDE": rede_row[COL_REDE_NSU],
                    "NÚM.NSU AUTCOM": aut_row[COL_AUT_NSU],
                    "Autorização REDE": rede_row[COL_REDE_AUT],
                    "Liberação AUTCOM": aut_row[COL_AUT_AUT],
                    "Valor bruto REDE": rede_row[COL_REDE_BRU],
                    "Valor bruto AUTCOM": aut_row[COL_AUT_BRU],
                    "Diferença bruto": diff_bru,
                    "Valor líquido REDE": rede_row[COL_REDE_LIQ],
                    "Valor líquido AUTCOM": aut_row[COL_AUT_LIQ],
                    "Diferença líquido": diff_liq,
                    "Pontuação match": best_score,
                }
            )
        else:
            matches.append(
                {
                    "_idx_rede": rede_row["_idx_rede"],
                    "_idx_aut": np.nan,
                    "Status geral": "❌ Não encontrado",
                    "NSU encontrado": "❌",
                    "Autorização encontrada": "❌",
                    "Valor bruto encontrado": "❌",
                    "Valor líquido encontrado": "❌",
                    "Data REDE": rede_row[COL_REDE_DATA],
                    "Data AUTCOM": pd.NaT,
                    "Diferença de datas (dias)": np.nan,
                    "NSU/CV REDE": rede_row[COL_REDE_NSU],
                    "NÚM.NSU AUTCOM": np.nan,
                    "Autorização REDE": rede_row[COL_REDE_AUT],
                    "Liberação AUTCOM": np.nan,
                    "Valor bruto REDE": rede_row[COL_REDE_BRU],
                    "Valor bruto AUTCOM": np.nan,
                    "Diferença bruto": np.nan,
                    "Valor líquido REDE": rede_row[COL_REDE_LIQ],
                    "Valor líquido AUTCOM": np.nan,
                    "Diferença líquido": np.nan,
                    "Pontuação match": np.nan,
                }
            )

    result = pd.DataFrame(matches)
    result = result.sort_values("_idx_rede").reset_index(drop=True)
    return result


def build_directions(df_found: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df_found.iterrows():
        acoes = []

        diff_data = row.get("Diferença de datas (dias)")
        diff_bru = row.get("Diferença bruto")
        diff_liq = row.get("Diferença líquido")

        if not pd.isna(diff_data) and int(diff_data) != 0:
            acoes.append(
                f"Alterar data e informar a data correta: REDE {format_date(row['Data REDE'])} | AUTCOM {format_date(row['Data AUTCOM'])}"
            )

        if not pd.isna(diff_bru) and abs(float(diff_bru)) > 0.01:
            acoes.append(
                f"Alterar valor bruto: REDE {brl(row['Valor bruto REDE'])} | AUTCOM {brl(row['Valor bruto AUTCOM'])} | Diferença {brl(diff_bru)}"
            )

        if not pd.isna(diff_liq) and abs(float(diff_liq)) > 0.01:
            acoes.append(
                f"Alterar valor líquido: REDE {brl(row['Valor líquido REDE'])} | AUTCOM {brl(row['Valor líquido AUTCOM'])} | Diferença {brl(diff_liq)}"
            )

        if acoes:
            rows.append(
                {
                    "NSU/CV REDE": row["NSU/CV REDE"],
                    "NÚM.NSU AUTCOM": row["NÚM.NSU AUTCOM"],
                    "Autorização REDE": row["Autorização REDE"],
                    "Liberação AUTCOM": row["Liberação AUTCOM"],
                    "Direcionamento": " | ".join(acoes),
                }
            )

    return pd.DataFrame(rows)


def display_money_cols(df: pd.DataFrame, money_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in money_cols:
        if c in out.columns:
            out[c] = out[c].apply(brl)
    return out


def render_df(df: pd.DataFrame, hide_index: bool = True, use_container_width: bool = True):
    st.dataframe(df, hide_index=hide_index, use_container_width=use_container_width)


# =========================
# UI
# =========================
st.title("Conciliação REDE x AUTCOM")
st.caption("Leitura da planilha, cruzamento robusto entre REDE e AUTCOM e geração de relatórios de divergência.")

with st.sidebar:
    st.header("Arquivo e filtros")

    uploaded = st.file_uploader(
        "Envie a planilha Excel",
        type=["xlsx"],
        help="O arquivo precisa conter as abas REDE e AUTCOM.",
    )

    use_local = st.checkbox("Usar arquivo local automaticamente", value=True)

default_file = Path(__file__).with_name("rede automação.xlsx")
file_bytes = None

if uploaded is not None:
    file_bytes = uploaded.getvalue()
elif use_local and default_file.exists():
    file_bytes = default_file.read_bytes()

if file_bytes is None:
    st.info("Envie a planilha Excel na lateral, ou coloque o arquivo 'rede automação.xlsx' na mesma pasta do app.py.")
    st.stop()

try:
    rede, aut = load_excel(file_bytes)
except Exception as e:
    st.error(f"Erro ao ler a planilha: {e}")
    st.stop()

# filtro de período com base na REDE
data_min = rede["_data_rede"].dropna().min()
data_max = rede["_data_rede"].dropna().max()

with st.sidebar:
    periodo = st.date_input(
        "Período (Data do recebimento - REDE)",
        value=(data_min.date(), data_max.date()) if pd.notna(data_min) and pd.notna(data_max) else None,
        min_value=data_min.date() if pd.notna(data_min) else None,
        max_value=data_max.date() if pd.notna(data_max) else None,
    )

if not periodo or len(periodo) != 2:
    st.warning("Selecione um período válido.")
    st.stop()

dt_ini = pd.to_datetime(periodo[0]).normalize()
dt_fim = pd.to_datetime(periodo[1]).normalize()

rede_f = rede[(rede["_data_rede"] >= dt_ini) & (rede["_data_rede"] <= dt_fim)].copy()

# AUTCOM filtrado de forma ampla para não perder matches
aut_f = aut.copy()
if aut["_data_aut"].notna().any():
    aut_f = aut[
        (aut["_data_aut"].isna()) |
        ((aut["_data_aut"] >= dt_ini - pd.Timedelta(days=35)) & (aut["_data_aut"] <= dt_fim + pd.Timedelta(days=35)))
    ].copy()

result = robust_match(rede_f, aut_f)

# métricas
qtd_total = len(result)
qtd_found = int((result["Status geral"] == "✅ Encontrado").sum())
qtd_not_found = int((result["Status geral"] == "❌ Não encontrado").sum())
total_bru_rede = rede_f["_vlr_bru"].sum(min_count=1)
total_liq_rede = rede_f["_vlr_liq"].sum(min_count=1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lançamentos REDE no período", f"{qtd_total:,}".replace(",", "."))
c2.metric("Encontrados", f"{qtd_found:,}".replace(",", "."))
c3.metric("Não encontrados", f"{qtd_not_found:,}".replace(",", "."))
c4.metric("Período", f"{dt_ini.strftime('%d/%m/%Y')} a {dt_fim.strftime('%d/%m/%Y')}")

st.subheader("1) Busca robusta REDE x AUTCOM")

main_cols = [
    "Status geral",
    "NSU encontrado",
    "Autorização encontrada",
    "Valor bruto encontrado",
    "Valor líquido encontrado",
    "Data REDE",
    "Data AUTCOM",
    "NSU/CV REDE",
    "NÚM.NSU AUTCOM",
    "Autorização REDE",
    "Liberação AUTCOM",
    "Valor bruto REDE",
    "Valor bruto AUTCOM",
    "Valor líquido REDE",
    "Valor líquido AUTCOM",
]

main_df = result[main_cols].copy()
main_df["Data REDE"] = main_df["Data REDE"].apply(format_date)
main_df["Data AUTCOM"] = main_df["Data AUTCOM"].apply(format_date)
main_df = display_money_cols(
    main_df,
    ["Valor bruto REDE", "Valor bruto AUTCOM", "Valor líquido REDE", "Valor líquido AUTCOM"],
)
render_df(main_df)

tb1, tb2 = st.columns(2)
tb1.metric("Soma valor bruto REDE", brl(total_bru_rede))
tb2.metric("Soma valor líquido REDE", brl(total_liq_rede))

found_df = result[result["Status geral"] == "✅ Encontrado"].copy()

# divergência de datas
st.subheader("2) Divergências de datas dentro dos encontrados")
div_datas = found_df[
    found_df["Diferença de datas (dias)"].notna() &
    (found_df["Diferença de datas (dias)"] != 0)
].copy()

if div_datas.empty:
    st.success("Nenhuma divergência de data encontrada dentro dos lançamentos encontrados.")
else:
    div_datas_show = div_datas[
        [
            "NSU/CV REDE",
            "NÚM.NSU AUTCOM",
            "Autorização REDE",
            "Liberação AUTCOM",
            "Data REDE",
            "Data AUTCOM",
            "Diferença de datas (dias)",
        ]
    ].copy()
    div_datas_show["Data REDE"] = div_datas_show["Data REDE"].apply(format_date)
    div_datas_show["Data AUTCOM"] = div_datas_show["Data AUTCOM"].apply(format_date)
    render_df(div_datas_show)

# divergência bruto
st.subheader("3) Divergências de valores brutos dentro dos encontrados")
div_bru = found_df[
    found_df["Diferença bruto"].notna() &
    (found_df["Diferença bruto"].abs() > 0.01)
].copy()

if div_bru.empty:
    st.success("Nenhuma divergência de valor bruto encontrada dentro dos lançamentos encontrados.")
else:
    div_bru_show = div_bru[
        [
            "NSU/CV REDE",
            "NÚM.NSU AUTCOM",
            "Autorização REDE",
            "Liberação AUTCOM",
            "Valor bruto REDE",
            "Valor bruto AUTCOM",
            "Diferença bruto",
        ]
    ].copy()
    div_bru_show = display_money_cols(
        div_bru_show,
        ["Valor bruto REDE", "Valor bruto AUTCOM", "Diferença bruto"],
    )
    render_df(div_bru_show)

# divergência líquido
st.subheader("4) Divergências de valores líquidos dentro dos encontrados")
div_liq = found_df[
    found_df["Diferença líquido"].notna() &
    (found_df["Diferença líquido"].abs() > 0.01)
].copy()

if div_liq.empty:
    st.success("Nenhuma divergência de valor líquido encontrada dentro dos lançamentos encontrados.")
else:
    div_liq_show = div_liq[
        [
            "NSU/CV REDE",
            "NÚM.NSU AUTCOM",
            "Autorização REDE",
            "Liberação AUTCOM",
            "Valor líquido REDE",
            "Valor líquido AUTCOM",
            "Diferença líquido",
        ]
    ].copy()
    div_liq_show = display_money_cols(
        div_liq_show,
        ["Valor líquido REDE", "Valor líquido AUTCOM", "Diferença líquido"],
    )
    render_df(div_liq_show)

# direcionamento final
st.subheader("5) Direcionamento do que precisa ser feito")
direcionamento = build_directions(found_df)

matched_aut_idxs = set(
    result.loc[result["_idx_aut"].notna(), "_idx_aut"].astype(int).tolist()
)
aut_not_found = aut_f[~aut_f["_idx_aut"].isin(matched_aut_idxs)].copy()

if direcionamento.empty:
    st.success("Nenhum ajuste necessário dentro dos lançamentos encontrados.")
else:
    render_df(direcionamento)

# bloco adicional: não encontrados
with st.expander("Ver lançamentos da REDE que não foram encontrados no AUTCOM"):
    not_found = result[result["Status geral"] == "❌ Não encontrado"].copy()
    if not not_found.empty:
        nf = not_found[
            [
                "Data REDE",
                "NSU/CV REDE",
                "Autorização REDE",
                "Valor bruto REDE",
                "Valor líquido REDE",
            ]
        ].copy()
        nf["Data REDE"] = nf["Data REDE"].apply(format_date)
        nf = display_money_cols(nf, ["Valor bruto REDE", "Valor líquido REDE"])
        render_df(nf)
    else:
        st.success("Todos os lançamentos da REDE tiveram algum match robusto no AUTCOM dentro do período.")

with st.expander("Ver lançamentos do AUTCOM que não foram encontrados na REDE"):
    if not aut_not_found.empty:
        anf = aut_not_found[
            [
                COL_AUT_DATA,
                COL_AUT_NSU,
                COL_AUT_AUT,
                COL_AUT_BRU,
                COL_AUT_LIQ,
            ]
        ].copy()
        anf.columns = [
            "Data AUTCOM",
            "NÚM.NSU AUTCOM",
            "Liberação AUTCOM",
            "Valor bruto AUTCOM",
            "Valor líquido AUTCOM",
        ]
        anf["Data AUTCOM"] = anf["Data AUTCOM"].apply(format_date)
        anf = display_money_cols(anf, ["Valor bruto AUTCOM", "Valor líquido AUTCOM"])
        render_df(anf)
    else:
        st.success("Todos os lançamentos do AUTCOM tiveram algum match robusto na REDE dentro do período.")

# downloads
st.subheader("6) Exportação")
export_buffer = io.BytesIO()

with pd.ExcelWriter(export_buffer, engine="openpyxl") as writer:
    main_export = result.copy()
    for c in ["Data REDE", "Data AUTCOM"]:
        if c in main_export.columns:
            main_export[c] = pd.to_datetime(main_export[c], errors="coerce")
    main_export.to_excel(writer, sheet_name="Conciliação", index=False)
    div_datas.to_excel(writer, sheet_name="Divergencias_Datas", index=False)
    div_bru.to_excel(writer, sheet_name="Divergencias_Bruto", index=False)
    div_liq.to_excel(writer, sheet_name="Divergencias_Liquido", index=False)
    direcionamento.to_excel(writer, sheet_name="Direcionamento", index=False)

st.download_button(
    "Baixar relatório em Excel",
    data=export_buffer.getvalue(),
    file_name="relatorio_conciliacao_rede_autcom.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
