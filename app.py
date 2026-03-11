"""
Memristor-LIF Neuron Simulator
================================
Simulation von 5 gekoppelten Memristor-basierten Leaky Integrate-and-Fire Neuronen

Physikalisches Modell:
  - HP-Memristor (Strukov et al., Nature 2008) mit Biolek-Fensterfunktion
  - LIF-Dynamik mit memristiver Leckkeitfähigkeit
  - Exponentielle synaptische Kopplung (all-to-all)
  - Rauschen: gaußsches weißes Rauschen (Langevin-Ansatz)

Autoren: Generiert mit Claude (Anthropic) für Prof. Henrich, Medizintechnik
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from io import BytesIO

# ─────────────────────────────────────────────────────────────────────────────
# Seite konfigurieren
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Memristor-LIF Neuron Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Memristor-LIF Neuron Simulator")
st.markdown(
    "Simulation von **5 gekoppelten Memristor-LIF-Neuronen** "
    "(HP-Modell · Biolek-Fenster · Exponentielle Synapsen)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Physikalische Konstanten & Einheitensystem
# ─────────────────────────────────────────────────────────────────────────────
# Zeiteinheit: ms | Spannung: mV | Strom: pA | Kapazität: pF | Widerstand: GΩ
# Leitfähigkeit: nS | Beweglichkeit: nm²/(V·ms)

N_NEURONS = 5
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
NEURON_LABELS = [f"Neuron {i+1}" for i in range(N_NEURONS)]

# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen: Memristor-Modell
# ─────────────────────────────────────────────────────────────────────────────

def biolek_window(w: np.ndarray, i: np.ndarray, p: int = 1) -> np.ndarray:
    """
    Biolek-Fensterfunktion zur Begrenzung des Memristorzustands auf [0,1].
    f(w, i) = 1 - (w - H(-i))^(2p)
    H(·) : Heaviside-Funktion
    """
    H = np.where(i >= 0, 1.0, 0.0)
    return 1.0 - (w - H) ** (2 * p)


def memristor_resistance(w: np.ndarray, R_on: float, R_off: float) -> np.ndarray:
    """
    Lineare Zustandsabhängigkeit des Widerstands.
    R(w) = w·R_on + (1-w)·R_off    [GΩ]
    """
    return w * R_on + (1.0 - w) * R_off


def memristor_conductance(w: np.ndarray, R_on: float, R_off: float) -> np.ndarray:
    """g(w) = 1/R(w)  [nS]"""
    return 1.0 / memristor_resistance(w, R_on, R_off)


# ─────────────────────────────────────────────────────────────────────────────
# Hauptsimulation (Euler-Vorwärtsdifferenz)
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    dt, T,
    C_m, V_rest, V_th, V_reset, tau_ref,
    R_on, R_off, mu_v, D_nm, p_window, w0_arr,
    I_ext_arr, noise_sigma,
    g_syn, tau_syn, delay_steps,
    seed
):
    """
    Euler-Integration des gekoppelten Memristor-LIF-Netzwerks.

    Dynamikgleichungen
    ------------------
    Membranpotential (LIF):
        C_m · dV_i/dt = g_mem(w_i)·(V_rest - V_i) + I_ext_i + I_syn_i + σ·ξ_i(t)

    Memristorzustand (HP-Modell):
        dw_i/dt = α · I_mem(t) · f(w_i)
        α = μ_v · R_on / D²        [1/(V·ms)] bei gegebenen Einheiten

    Synaptische Aktivierungsvariable:
        ds_j/dt = -s_j / τ_syn + Σ_k δ(t - t_k^spike)
        I_syn_i = g_syn · Σ_{j≠i} s_j(t - τ_delay) · (E_syn - V_i)
    """
    rng = np.random.default_rng(seed)

    n_steps = int(T / dt)
    D_m = D_nm * 1e-9          # nm → m
    alpha = mu_v * R_on / D_m**2  # Memristor-Mobilitätsfaktor  [1/(V·s)] → skalieren

    # Zustandsvektoren
    V     = np.full(N_NEURONS, V_rest, dtype=float)        # Membranpotential [mV]
    w     = w0_arr.copy().astype(float)                    # Memristorzustand [0,1]
    s     = np.zeros(N_NEURONS)                            # Synaptische Variable
    ref   = np.zeros(N_NEURONS, dtype=int)                 # Refraktär-Zähler [Schritte]

    # Aufzeichnungspuffer
    V_rec = np.zeros((N_NEURONS, n_steps))
    w_rec = np.zeros((N_NEURONS, n_steps))
    R_rec = np.zeros((N_NEURONS, n_steps))
    spike_rec = [[] for _ in range(N_NEURONS)]

    # Synaptischer Delay-Puffer (Ringpuffer)
    buf_len = max(delay_steps + 1, 1)
    s_buf   = np.zeros((N_NEURONS, buf_len))   # Ringpuffer für synaptische Variablen

    E_syn = 0.0   # exzitatorisches Umkehrpotential [mV]

    for step in range(n_steps):
        t = step * dt

        # --- Memristiver Leitwert aller Neuronen ---
        g_mem = memristor_conductance(w, R_on, R_off)   # [nS]
        R_mem = memristor_resistance(w, R_on, R_off)    # [GΩ]

        # --- Memriststrom (für dw/dt) ---
        I_leak = g_mem * (V_rest - V) * 1e-3   # [pA] → [nA] für α-Skalierung
        # Vereinfachung: I_mem ≈ I_leak (dominanter Strom)

        # --- Synaptische Ströme (mit Verzögerung) ---
        buf_idx = step % buf_len
        s_delayed = s_buf[:, buf_idx]
        I_syn = np.zeros(N_NEURONS)
        for i in range(N_NEURONS):
            for j in range(N_NEURONS):
                if i != j:
                    I_syn[i] += g_syn * s_delayed[j] * (E_syn - V[i])

        # --- Gaußsches Rauschen ---
        noise = rng.normal(0, noise_sigma, N_NEURONS) * np.sqrt(dt)

        # --- Membranpotential-Update (nur wenn nicht refraktär) ---
        for i in range(N_NEURONS):
            if ref[i] > 0:
                V[i] = V_reset
                ref[i] -= 1
            else:
                dV = (g_mem[i] * (V_rest - V[i]) + I_ext_arr[i] + I_syn[i]) / C_m * dt
                V[i] += dV + noise[i] / C_m

                # Spike-Detektion
                if V[i] >= V_th:
                    V[i] = V_reset
                    ref[i] = int(tau_ref / dt)
                    spike_rec[i].append(t)
                    s[i] += 1.0   # Präsynaptische Aktivierung

        # --- Memristorzustand-Update ---
        I_mem_norm = I_leak * 1e3   # zurück zu [pA] für Konsistenz
        # α in biologisch sinnvolle Einheiten skalieren
        alpha_scaled = alpha * 1e-12 * dt  # dt in ms → s
        dw = alpha_scaled * I_mem_norm * biolek_window(w, I_mem_norm, p=p_window)
        w = np.clip(w + dw, 0.0, 1.0)

        # --- Synaptische Variable ---
        ds = -s / tau_syn * dt
        s = np.maximum(s + ds, 0.0)
        s_buf[:, (step + 1) % buf_len] = s

        # --- Aufzeichnung ---
        V_rec[:, step] = V
        w_rec[:, step] = w
        R_rec[:, step] = R_mem

    t_arr = np.arange(n_steps) * dt
    return t_arr, V_rec, w_rec, R_rec, spike_rec


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Parameter
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⏱ Simulation")
    dt  = st.slider("Zeitschritt dt (ms)", 0.01, 0.5, 0.05, 0.01, format="%.2f")
    T   = st.slider("Gesamtzeit T (ms)",  100, 3000, 800, 50)
    seed = st.number_input("Zufalls-Seed", 0, 9999, 42, 1)

    st.divider()
    st.header("⚡ LIF-Parameter")
    C_m      = st.slider("Membrankapazität C_m (pF)", 10.0, 500.0, 100.0, 10.0)
    V_rest   = st.slider("Ruhepotential V_rest (mV)", -80, -55, -65, 1)
    V_th     = st.slider("Schwellenpotential V_th (mV)", -55, -40, -50, 1)
    V_reset  = st.slider("Reset-Potential V_reset (mV)", -85, -60, -70, 1)
    tau_ref  = st.slider("Refraktärzeit τ_ref (ms)", 0.5, 10.0, 2.0, 0.5)

    st.divider()
    st.header("🔴 Memristor-Parameter")
    R_on     = st.slider("R_on (GΩ)", 0.01, 1.0, 0.1, 0.01, format="%.2f")
    R_off    = st.slider("R_off (GΩ)", 1.0, 50.0, 16.0, 0.5)
    mu_v     = st.slider("Ionenmobilität μ_v (×10⁻¹⁴ m²/Vs)", 0.1, 20.0, 1.0, 0.1) * 1e-14
    D_nm     = st.slider("Membrandicke D (nm)", 1.0, 30.0, 10.0, 0.5)
    p_window = st.select_slider("Biolek-Exponent p", [1, 2, 3, 4, 5], 1)

    st.divider()
    st.header("🔌 Externer Strom (pA)")
    I_ext_arr = np.array([
        st.slider(f"I_ext Neuron {i+1}", 0.0, 2000.0, [800., 900., 1000., 850., 950.][i], 10.0)
        for i in range(N_NEURONS)
    ])
    noise_sigma = st.slider("Rauschstärke σ (pA·√ms)", 0.0, 500.0, 50.0, 10.0)

    st.divider()
    st.header("🔗 Synaptische Kopplung")
    g_syn   = st.slider("Kopplungsleitwert g_syn (nS)", 0.0, 5.0, 0.3, 0.05, format="%.2f")
    tau_syn = st.slider("Synaptische Zeitkonstante τ_syn (ms)", 1.0, 30.0, 5.0, 0.5)
    delay_ms= st.slider("Axonale Verzögerung (ms)", 0.1, 10.0, 1.0, 0.1)

    st.divider()
    st.header("🧬 Anfangszustand w₀")
    w0_arr = np.array([
        st.slider(f"w₀ Neuron {i+1}", 0.0, 1.0, [0.3, 0.5, 0.7, 0.4, 0.6][i], 0.05)
        for i in range(N_NEURONS)
    ])

    run_btn = st.button("▶ Simulation starten", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Formelanzeige (Hauptbereich, vor der Simulation)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📐 Physikalisches Modell (Gleichungen)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(r"""
**LIF-Dynamik mit memristivem Leck:**

$$C_m \frac{dV_i}{dt} = g_\mathrm{mem}(w_i)(V_\mathrm{rest}-V_i) + I_\mathrm{ext,i} + I_\mathrm{syn,i} + \sigma\,\xi_i(t)$$

**Spike:** Falls $V_i \geq V_\mathrm{th}$ → $V_i \leftarrow V_\mathrm{reset}$, Refraktärperiode $\tau_\mathrm{ref}$

**Synaptische Kopplung (exponentiell):**

$$\frac{ds_j}{dt} = -\frac{s_j}{\tau_\mathrm{syn}} + \sum_k \delta(t-t_k^\mathrm{sp})$$

$$I_\mathrm{syn,i} = g_\mathrm{syn} \sum_{j \neq i} s_j(t-\tau_d)(E_\mathrm{syn}-V_i)$$
""")
    with col2:
        st.markdown(r"""
**HP-Memristor (Strukov et al. 2008):**

$$R(w_i) = w_i R_\mathrm{on} + (1-w_i) R_\mathrm{off}, \quad w_i \in [0,1]$$

$$g_\mathrm{mem}(w_i) = \frac{1}{R(w_i)}$$

**Zustandsgleichung mit Biolek-Fenster:**

$$\frac{dw_i}{dt} = \alpha \cdot I_\mathrm{mem}(t) \cdot f(w_i, I)$$

$$\alpha = \frac{\mu_v R_\mathrm{on}}{D^2}, \quad f(w,I) = 1-(w-H(-I))^{2p}$$
""")

# ─────────────────────────────────────────────────────────────────────────────
# Simulation ausführen
# ─────────────────────────────────────────────────────────────────────────────

if run_btn or "sim_results" in st.session_state:

    if run_btn:
        delay_steps = max(1, int(delay_ms / dt))
        with st.spinner("Simuliere Netzwerk…"):
            t_arr, V_rec, w_rec, R_rec, spike_rec = run_simulation(
                dt=dt, T=T,
                C_m=C_m, V_rest=V_rest, V_th=V_th, V_reset=V_reset, tau_ref=tau_ref,
                R_on=R_on, R_off=R_off, mu_v=mu_v, D_nm=D_nm,
                p_window=p_window, w0_arr=w0_arr,
                I_ext_arr=I_ext_arr, noise_sigma=noise_sigma,
                g_syn=g_syn, tau_syn=tau_syn, delay_steps=delay_steps,
                seed=int(seed)
            )
        st.session_state["sim_results"] = (t_arr, V_rec, w_rec, R_rec, spike_rec)
        st.session_state["sim_params"] = dict(V_th=V_th, V_reset=V_reset, V_rest=V_rest)
    else:
        t_arr, V_rec, w_rec, R_rec, spike_rec = st.session_state["sim_results"]

    # ─────────────────────────────────────────────────────────────────────────
    # Kennzahlen
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📊 Spike-Statistik")
    cols = st.columns(N_NEURONS)
    for i, col in enumerate(cols):
        n_sp = len(spike_rec[i])
        rate = n_sp / (T * 1e-3) if T > 0 else 0  # Hz
        isi_cv = 0.0
        if n_sp > 1:
            isis = np.diff(spike_rec[i])
            isi_cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
        col.metric(
            label=f"**Neuron {i+1}**",
            value=f"{n_sp} Spikes",
            delta=f"{rate:.1f} Hz  |  CV={isi_cv:.2f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Hauptplot: 4 Panels
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(4, 1, hspace=0.45,
                             left=0.07, right=0.97, top=0.94, bottom=0.07)

    ax_V   = fig.add_subplot(gs[0])
    ax_sp  = fig.add_subplot(gs[1], sharex=ax_V)
    ax_w   = fig.add_subplot(gs[2], sharex=ax_V)
    ax_R   = fig.add_subplot(gs[3], sharex=ax_V)

    for ax in [ax_V, ax_sp, ax_w, ax_R]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="major", color="#2a2a5a", linewidth=0.5)
        ax.grid(True, which="minor", color="#1e1e3a", linewidth=0.3)

    # Panel 1: Membranpotentiale
    ax_V.set_title("Membranpotentiale V(t)", fontsize=10, fontweight="bold")
    ax_V.set_ylabel("V  (mV)", fontsize=9)
    for i in range(N_NEURONS):
        ax_V.plot(t_arr, V_rec[i], color=COLORS[i], lw=0.7, alpha=0.9,
                  label=NEURON_LABELS[i])
    ax_V.axhline(V_th, color="white", lw=0.8, ls="--", alpha=0.4, label=f"V_th = {V_th} mV")
    ax_V.axhline(V_rest, color="#aaaaaa", lw=0.5, ls=":", alpha=0.3)
    ax_V.legend(fontsize=7, loc="upper right", framealpha=0.2,
                labelcolor="white", facecolor="#1a1a2e")

    # Panel 2: Spike-Raster
    ax_sp.set_title("Spike-Raster", fontsize=10, fontweight="bold")
    ax_sp.set_ylabel("Neuron", fontsize=9)
    for i in range(N_NEURONS):
        if spike_rec[i]:
            ax_sp.vlines(spike_rec[i], i + 0.5, i + 1.5,
                         color=COLORS[i], lw=1.2, alpha=0.9)
    ax_sp.set_ylim(0.3, N_NEURONS + 0.7)
    ax_sp.set_yticks(range(1, N_NEURONS + 1))
    ax_sp.set_yticklabels([f"N{i+1}" for i in range(N_NEURONS)], fontsize=8)

    # Panel 3: Memristorzustände
    ax_w.set_title("Memristorzustand w(t)", fontsize=10, fontweight="bold")
    ax_w.set_ylabel("w  (0…1)", fontsize=9)
    for i in range(N_NEURONS):
        ax_w.plot(t_arr, w_rec[i], color=COLORS[i], lw=0.8, alpha=0.9)
    ax_w.set_ylim(-0.05, 1.05)
    ax_w.axhline(0, color="#555577", lw=0.5)
    ax_w.axhline(1, color="#555577", lw=0.5)

    # Panel 4: Memristiver Widerstand
    ax_R.set_title("Memristiver Widerstand R(t)", fontsize=10, fontweight="bold")
    ax_R.set_ylabel("R  (GΩ)", fontsize=9)
    ax_R.set_xlabel("Zeit  (ms)", fontsize=9)
    for i in range(N_NEURONS):
        ax_R.plot(t_arr, R_rec[i], color=COLORS[i], lw=0.8, alpha=0.9)

    fig.suptitle(
        f"Memristor-LIF Netzwerk  ·  N={N_NEURONS}  ·  T={T} ms  ·  dt={dt} ms",
        color="white", fontsize=12, fontweight="bold", y=0.98
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Sekundärplot: ISI-Histogramme + Phasenporträt
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("🔬 ISI-Analyse & Phasenporträt")
    col_hist, col_phase = st.columns([3, 2])

    with col_hist:
        fig2, axes = plt.subplots(1, N_NEURONS, figsize=(13, 3),
                                   facecolor="#1a1a2e", sharey=False)
        for i in range(N_NEURONS):
            ax = axes[i]
            ax.set_facecolor("#16213e")
            for spine in ax.spines.values():
                spine.set_color("#444466")
            ax.tick_params(colors="white", labelsize=7)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

            if len(spike_rec[i]) > 2:
                isis = np.diff(spike_rec[i])
                ax.hist(isis, bins=min(20, len(isis)), color=COLORS[i],
                        alpha=0.8, edgecolor="none")
                ax.set_xlabel("ISI (ms)", fontsize=7)
                ax.set_title(f"N{i+1}\nμ={np.mean(isis):.1f} ms", fontsize=8)
            else:
                ax.text(0.5, 0.5, "< 3 Spikes", color="white",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_title(f"N{i+1}", fontsize=8)
            if i == 0:
                ax.set_ylabel("Anzahl", fontsize=7)

        fig2.suptitle("ISI-Histogramme", color="white", fontsize=10, y=1.02)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with col_phase:
        neuron_idx = st.selectbox("Neuron für Phasenporträt", range(1, 6),
                                   format_func=lambda x: f"Neuron {x}") - 1
        fig3, ax3 = plt.subplots(figsize=(4.5, 4), facecolor="#1a1a2e")
        ax3.set_facecolor("#16213e")
        for spine in ax3.spines.values():
            spine.set_color("#444466")
        ax3.tick_params(colors="white", labelsize=8)
        ax3.xaxis.label.set_color("white")
        ax3.yaxis.label.set_color("white")
        ax3.title.set_color("white")

        sc = ax3.scatter(
            w_rec[neuron_idx, ::5], V_rec[neuron_idx, ::5],
            c=t_arr[::5], cmap="plasma", s=0.5, alpha=0.7, linewidths=0
        )
        cbar = fig3.colorbar(sc, ax=ax3, shrink=0.85)
        cbar.set_label("t (ms)", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax3.set_xlabel("Memristorzustand w", fontsize=9)
        ax3.set_ylabel("V  (mV)", fontsize=9)
        ax3.set_title(f"Phasenporträt — Neuron {neuron_idx+1}", fontsize=9)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    # ─────────────────────────────────────────────────────────────────────────
    # Feuerrate-Balkendiagramm
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📈 Mittlere Feuerrate & Memristorzustand (Ende)")
    col_bar1, col_bar2 = st.columns(2)

    with col_bar1:
        fig4, ax4 = plt.subplots(figsize=(5, 3), facecolor="#1a1a2e")
        ax4.set_facecolor("#16213e")
        for spine in ax4.spines.values():
            spine.set_color("#444466")
        ax4.tick_params(colors="white")
        ax4.xaxis.label.set_color("white")
        ax4.yaxis.label.set_color("white")

        rates = [len(spike_rec[i]) / (T * 1e-3) for i in range(N_NEURONS)]
        bars = ax4.bar(NEURON_LABELS, rates, color=COLORS, edgecolor="none", alpha=0.85)
        ax4.set_ylabel("Feuerrate (Hz)", color="white")
        ax4.set_title("Mittlere Feuerrate", color="white", fontsize=10)
        for bar, rate in zip(bars, rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{rate:.1f}", ha="center", va="bottom", color="white", fontsize=8)
        plt.xticks(rotation=20, ha="right")
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

    with col_bar2:
        fig5, ax5 = plt.subplots(figsize=(5, 3), facecolor="#1a1a2e")
        ax5.set_facecolor("#16213e")
        for spine in ax5.spines.values():
            spine.set_color("#444466")
        ax5.tick_params(colors="white")
        ax5.xaxis.label.set_color("white")
        ax5.yaxis.label.set_color("white")

        w_end = w_rec[:, -1]
        bars2 = ax5.bar(NEURON_LABELS, w_end, color=COLORS, edgecolor="none", alpha=0.85)
        ax5.set_ylim(0, 1)
        ax5.set_ylabel("w (Endwert)", color="white")
        ax5.set_title("Memristorzustand w(T)", color="white", fontsize=10)
        for bar, wv in zip(bars2, w_end):
            ax5.text(bar.get_x() + bar.get_width()/2, wv + 0.02,
                     f"{wv:.3f}", ha="center", va="bottom", color="white", fontsize=8)
        plt.xticks(rotation=20, ha="right")
        fig5.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💾 Daten exportieren")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        # Zeitreihen als CSV
        n_steps = len(t_arr)
        csv_header = "t_ms," + ",".join([f"V_N{i+1}_mV,w_N{i+1},R_N{i+1}_GOhm"
                                          for i in range(N_NEURONS)])
        rows = [csv_header]
        for k in range(n_steps):
            row = f"{t_arr[k]:.3f}"
            for i in range(N_NEURONS):
                row += f",{V_rec[i,k]:.4f},{w_rec[i,k]:.6f},{R_rec[i,k]:.4f}"
            rows.append(row)
        csv_data = "\n".join(rows)
        st.download_button(
            "⬇ Zeitreihen (CSV)",
            data=csv_data,
            file_name="memristor_lif_timeseries.csv",
            mime="text/csv",
            use_container_width=True
        )

    with exp_col2:
        # Spike-Zeiten als CSV
        spike_rows = ["neuron,spike_time_ms"]
        for i in range(N_NEURONS):
            for t_sp in spike_rec[i]:
                spike_rows.append(f"{i+1},{t_sp:.3f}")
        spike_csv = "\n".join(spike_rows)
        st.download_button(
            "⬇ Spike-Zeiten (CSV)",
            data=spike_csv,
            file_name="memristor_lif_spikes.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👈  Parameter in der Seitenleiste einstellen und **▶ Simulation starten** drücken.")

# ─────────────────────────────────────────────────────────────────────────────
# Fußzeile
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Referenzen:** Strukov et al. (2008) *Nature* 453:80–83 · "
    "Biolek et al. (2009) *RADIOENGINEERING* 18:210–214 · "
    "Mahowald & Douglas (1991) *Nature* 354:515–518  |  "
    "Medizintechnik · Prof. Henrich"
)
