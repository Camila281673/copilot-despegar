"""
app.py — Prototipo "Recover the Long Tail" (Streamlit)
================================================================================
Copilot de priorización para Account Managers de Despegar (Travel Partners).

La IA NO reemplaza al equipo: le dice **a quién llamar primero y qué decir**.

Tres vistas:
    1. Dashboard        — el problema y los segmentos de un vistazo
    2. Cola del AM       — los hoteles del target priorizados por oportunidad
    3. Copilot por hotel — diagnóstico + acción + mensaje listo para enviar

Toda la lógica de scoring/segmentación/recomendación viene de `motor_recover.py`,
el MISMO módulo que usa el análisis → los números de la app == los del notebook.

Cómo correr:
    pip install -r requirements.txt
    python preparar_datos.py        # genera hoteles_enriquecidos.csv (si no existe)
    streamlit run app.py

La parte de IA generativa (redactar el mensaje con Claude) es OPCIONAL: por
defecto la app funciona con reglas/plantillas, sin API key.
================================================================================
"""

import os
import pandas as pd
import streamlit as st

import motor_recover as motor

st.set_page_config(page_title="Recover the Long Tail", page_icon="🏨", layout="wide")

ARCHIVO_SCORED = "hoteles_enriquecidos.csv"
ARCHIVO_ORIGINAL = "dataset_hoteles_ejercicio.xlsx"


# ==============================================================================
# Carga de datos (cacheada para no recalcular en cada interacción)
# ==============================================================================
@st.cache_data
def cargar():
    """
    Carga el dataset ya procesado (hoteles_enriquecidos.csv). Si no existe, lo genera
    al vuelo desde el Excel con el motor. Así la app funciona aunque el usuario
    no haya corrido 'preparar_datos.py' antes.
    """
    if os.path.exists(ARCHIVO_SCORED):
        df = pd.read_csv(ARCHIVO_SCORED)
        # Si faltara alguna columna calculada, reprocesamos por las dudas.
        if "segmento" not in df.columns or "engagement_score" not in df.columns:
            df = motor.procesar(df)
        return df
    if os.path.exists(ARCHIVO_ORIGINAL):
        return motor.procesar_desde_archivo(ARCHIVO_ORIGINAL)
    return None


def crear_cliente_api():
    """
    Crea el cliente de Anthropic SOLO si hay una API key disponible
    (en st.secrets o en la variable de entorno ANTHROPIC_API_KEY).
    Si no hay key o no está instalada la librería, devuelve None → la app usa reglas.
    La key NUNCA se hardcodea ni se muestra.
    """
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


df = cargar()
if df is None:
    st.error(
        "No se encontró ni 'hoteles_scored.csv' ni 'dataset_hoteles_ejercicio.xlsx'. "
        "Corré primero `python preparar_datos.py` o subí el dataset a la carpeta."
    )
    st.stop()


# ==============================================================================
# Barra lateral — navegación
# ==============================================================================
st.sidebar.title("🏨 Recover the Long Tail")
st.sidebar.caption("Copilot de priorización · Travel Partners · Despegar")
vista = st.sidebar.radio(
    "Vista",
    ["1 · Dashboard", "2 · Cola del Account Manager", "3 · Copilot por hotel"],
)

cliente_api = crear_cliente_api()
estado_ia = "🟢 Claude API conectada" if cliente_api else "⚪ Modo reglas (sin API key)"
st.sidebar.markdown("---")
st.sidebar.caption(f"Motor de recomendación: {estado_ia}")


# ==============================================================================
# VISTA 1 — DASHBOARD
# ==============================================================================
if vista.startswith("1"):
    st.title("El 73% de los hoteles no vendió nada en 90 días")
    st.caption("El problema del long tail, de un vistazo.")

    total = len(df)
    ceros = int((df["reservas_90d"] == 0).sum())
    reservas_tot = int(df["reservas_90d"].sum())
    orden = df.sort_values("reservas_90d", ascending=False)
    top20 = int(round(0.20 * total))
    pct_top20 = orden["reservas_90d"].head(top20).sum() / reservas_tot

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reservas en el top 20%", f"{pct_top20:.0%}")
    c2.metric("Hoteles con 0 reservas", f"{ceros}")
    c3.metric("Capacidad del equipo", "150")
    c4.metric("Hoteles totales", f"{total}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("El bajo rendimiento tiene causas distintas")
        conteo = df["segmento"].value_counts()
        st.bar_chart(conteo, color="#1f6f8b", horizontal=True)
        st.caption(f"Target: **{motor.SEGMENTO_OBJETIVO}** ({int((df['segmento']==motor.SEGMENTO_OBJETIVO).sum())} hoteles).")

    with col_b:
        st.subheader("Impacto potencial de recuperar el target")
        imp = motor.estimar_impacto(df)
        impacto_df = pd.DataFrame({
            "escenario": ["Conservador", "Base", "Techo teórico"],
            "reservas_adicionales": [imp["escenario_conservador"], imp["escenario_base"], imp["techo_reservas"]],
        }).set_index("escenario")
        st.bar_chart(impacto_df, color="#e07a5f")
        st.caption(imp["supuestos"])

    st.markdown("---")
    st.subheader("Perfil de cada segmento")
    perfil = df.groupby("segmento").agg(
        hoteles=("hotel_id", "size"),
        clicks_prom=("clicks_90d", "mean"),
        disponibilidad=("disponibilidad_pct", "mean"),
        dias_tarifa=("dias_sin_actualizar_tarifas", "mean"),
        pct_sin_fotos=("tiene_fotos", lambda s: (~s).mean()),
        rating=("rating_promedio", "mean"),
        reservas=("reservas_90d", "sum"),
    ).round(2).sort_values("hoteles", ascending=False)
    st.dataframe(perfil, use_container_width=True)


# ==============================================================================
# VISTA 2 — COLA DEL ACCOUNT MANAGER
# ==============================================================================
elif vista.startswith("2"):
    st.title(f"A quién llamar primero: {motor.SEGMENTO_OBJETIVO}")
    st.caption("Hoteles que reciben tráfico pero no convierten, ordenados por oportunidad.")

    objetivo = df[df["segmento"] == motor.SEGMENTO_OBJETIVO].copy()
    if objetivo.empty:
        st.info("No hay hoteles en el segmento objetivo con los datos actuales.")
        st.stop()

    # "Oportunidad" = clicks que ya reciben (demanda existente sin convertir).
    # A más clicks desperdiciados, mayor prioridad de contacto.
    objetivo = objetivo.sort_values("clicks_90d", ascending=False)

    # Causa raíz principal por hotel (la primera que detecta el motor).
    def causa_principal(fila):
        causas = motor.diagnosticar_causa(fila)
        return causas[0]["titulo"] if causas else "Sin causa evidente"

    objetivo["causa_principal"] = objetivo.apply(causa_principal, axis=1)

    n = st.slider("¿Cuántos hoteles mostrar?", 5, len(objetivo), min(20, len(objetivo)))
    columnas = ["hotel_id", "nombre_hotel", "ciudad", "pais", "clicks_90d",
                "conversion_rate", "engagement_score", "causa_principal"]
    st.dataframe(objetivo[columnas].head(n), use_container_width=True, hide_index=True)

    st.caption(
        f"{len(objetivo)} hoteles en el target · {int(objetivo['clicks_90d'].sum())} clicks/90d "
        "de demanda que hoy no se convierte."
    )


# ==============================================================================
# VISTA 3 — COPILOT POR HOTEL
# ==============================================================================
else:
    st.title("Copilot por hotel")
    st.caption("La IA prioriza y sugiere; el Account Manager negocia y cierra.")

    objetivo = df[df["segmento"] == motor.SEGMENTO_OBJETIVO].copy()
    if objetivo.empty:
        st.info("No hay hoteles en el segmento objetivo con los datos actuales.")
        st.stop()
    objetivo = objetivo.sort_values("clicks_90d", ascending=False)

    # Selector de hotel
    etiquetas = objetivo["nombre_hotel"] + " — " + objetivo["ciudad"]
    eleccion = st.selectbox("Elegí un hotel del target:", etiquetas.tolist())
    fila = objetivo[etiquetas == eleccion].iloc[0]

    usar_ia = st.toggle(
        "Redactar el mensaje con IA (Claude)",
        value=False,
        help="Si está apagado o no hay API key, se usa una plantilla por reglas.",
    )

    # Métricas del hotel elegido
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clicks (90d)", int(fila["clicks_90d"]))
    c2.metric("Reservas (90d)", int(fila["reservas_90d"]))
    c3.metric("Fotos", int(fila["cantidad_fotos"]))
    c4.metric("Engagement", f"{fila['engagement_score']:.0f}")

    st.markdown("---")
    col_diag, col_msg = st.columns(2)

    # Diagnóstico de causas
    with col_diag:
        st.subheader("🔍 Diagnóstico (automático)")
        causas = motor.diagnosticar_causa(fila)
        if not causas:
            st.write("No se detectaron causas evidentes.")
        for c in causas:
            st.markdown(f"**{c['titulo']}**  \n{c['detalle']}")

    # Recomendación + mensaje
    with col_msg:
        st.subheader("✍️ Acción recomendada + mensaje")
        cliente = cliente_api if usar_ia else None
        rec = motor.recomendar(fila, cliente_api=cliente)

        st.markdown("**Acciones a pedir al hotelero:**")
        for a in rec["acciones"]:
            st.markdown(f"- {a}")

        st.markdown(f"**Borrador de mensaje** _(fuente: {rec['fuente']})_:")
        st.text_area("Mensaje", rec["mensaje"], height=240, label_visibility="collapsed")

        if usar_ia and cliente is None:
            st.info("No hay API key configurada → se usó la plantilla por reglas.")

    st.markdown("---")
    st.caption(
        "Flujo: la IA detecta la causa y propone la acción + el mensaje. "
        "El Account Manager **revisa, llama y cierra** el cambio con el hotel."
    )
