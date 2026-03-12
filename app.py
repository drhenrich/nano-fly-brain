"""
Memristor-LIF Neuron Simulator + Motorneuron
=============================================
5 gekoppelte Memristor-LIF-Neuronen steuern ein nachgeschaltetes Motorneuron.

Physikalisches Modell:
  - HP-Memristor (Strukov et al., Nature 2008) + Biolek-Fensterfunktion
  - LIF-Dynamik mit memristiver Leckleitfähigkeit (Interneuronen)
  - Standard-LIF-Motorneuron (klassisch, ohne Memristor)
  - Exponentielle synaptische Kopplung (all-to-all intern + Ensemble→Motor)
  - Motorausgangssignal: geglättete Feuerrate → Motorbefehl [0…1]

Autoren: Generiert mit Claude (Anthropic) für Prof. Dr. Dietmar Henrich,
         Physik & Medizintechnik / NanoFlyBrain-Projekt
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# Seite konfigurieren
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Memristor-LIF + Motorneuron",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Memristor-LIF Simulator  ➜  🦾 Motorneuron")
st.markdown(
    "**5 Memristor-LIF-Interneuronen** (HP-Modell · Biolek-Fenster) "
    "steuern ein nachgeschaltetes **Standard-LIF-Motorneuron** "
    "– Motorbefehl als geglättete Feuerrate."
)

# ─────────────────────────────────────────────────────────────────────────────
# Konstanten
# ─────────────────────────────────────────────────────────────────────────────
# Einheiten: ms | mV | pA | pF | GΩ | nS
N_NEURONS  = 5
COLORS     = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
COLOR_MOT  = "#00e5ff"   # Cyan für Motorneuron
NEURON_LABELS = [f"N{i+1}" for i in range(N_NEURONS)]

# ─────────────────────────────────────────────────────────────────────────────
# Memristor-Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def biolek_window(w, i, p=1):
    """f(w,i) = 1 - (w - H(-i))^(2p)   [Biolek et al. 2009]"""
    H = np.where(i >= 0, 1.0, 0.0)
    return 1.0 - (w - H) ** (2 * p)

def R_mem(w, R_on, R_off):
    """R(w) = w·R_on + (1-w)·R_off  [HP-Modell, Strukov 2008]"""
    return w * R_on + (1.0 - w) * R_off

def g_mem(w, R_on, R_off):
    return 1.0 / R_mem(w, R_on, R_off)

# ─────────────────────────────────────────────────────────────────────────────
# Hauptsimulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    dt, T,
    # Interneuronen (Memristor-LIF)
    C_m, V_rest, V_th, V_reset, tau_ref,
    R_on, R_off, mu_v, D_nm, p_window, w0_arr,
    I_ext_arr, noise_sigma,
    g_syn, tau_syn, delay_steps,
    # Motorneuron (Standard-LIF)
    C_mot, V_rest_mot, V_th_mot, V_reset_mot, tau_ref_mot,
    R_leak_mot,          # Leckkwiderstand Motorneuron [GΩ]
    g_mot_arr,           # Kopplungsstärke N_i → Motor [nS], Länge N_NEURONS
    tau_syn_mot,         # Synaptische Zeitkonstante Motor [ms]
    noise_mot,           # Rauschen Motorneuron [pA·√ms]
    tau_smooth,          # Glättungszeitkonstante Motorausgangssignal [ms]
    seed
):
    """
    Euler-Integration:
      - N_NEURONS Memristor-LIF-Interneuronen (all-to-all Kopplung)
      - 1 Standard-LIF-Motorneuron (getrieben durch Ensemble-Aktivität)

    Motorneuron-Gleichung:
      C_mot · dV_mot/dt = (V_rest_mot - V_mot)/R_leak_mot
                         + Σ_i g_mot_i · s_i(t) · (E_syn - V_mot)
                         + σ_mot · ξ(t)

    Motorausgangssignal:
      r(t) = Σ_k exp(-(t - t_k)/τ_smooth) · H(t - t_k)   [Hz-normiert]
    """
    rng = np.random.default_rng(seed)
    n_steps  = int(T / dt)
    D_m      = D_nm * 1e-9
    alpha    = mu_v * R_on / D_m**2

    # ── Interneuronen-Zustand ─────────────────────────────────────────────────
    V    = np.full(N_NEURONS, V_rest, dtype=float)
    w    = w0_arr.copy().astype(float)
    s    = np.zeros(N_NEURONS)
    ref  = np.zeros(N_NEURONS, dtype=int)

    buf_len = max(delay_steps + 1, 1)
    s_buf   = np.zeros((N_NEURONS, buf_len))
    E_syn   = 0.0

    # ── Motorneuron-Zustand ───────────────────────────────────────────────────
    V_mot    = V_rest_mot
    ref_mot  = 0
    s_mot    = np.zeros(N_NEURONS)   # Synaptische Variablen Motor-Eingänge

    # ── Aufzeichnung ──────────────────────────────────────────────────────────
    V_rec      = np.zeros((N_NEURONS, n_steps))
    w_rec      = np.zeros((N_NEURONS, n_steps))
    R_rec      = np.zeros((N_NEURONS, n_steps))
    spike_rec  = [[] for _ in range(N_NEURONS)]

    V_mot_rec  = np.zeros(n_steps)
    spike_mot  = []

    for step in range(n_steps):
        t = step * dt

        # ── Memristiver Leitwert ──────────────────────────────────────────────
        gm   = g_mem(w, R_on, R_off)
        Rm   = R_mem(w, R_on, R_off)
        I_lk = gm * (V_rest - V) * 1e-3

        # ── Interneuron-Synapsen (mit Delay) ──────────────────────────────────
        buf_idx   = step % buf_len
        s_delayed = s_buf[:, buf_idx]
        I_syn_int = np.zeros(N_NEURONS)
        for i in range(N_NEURONS):
            for j in range(N_NEURONS):
                if i != j:
                    I_syn_int[i] += g_syn * s_delayed[j] * (E_syn - V[i])

        # ── Rauschen ──────────────────────────────────────────────────────────
        noise_int = rng.normal(0, noise_sigma, N_NEURONS) * np.sqrt(dt)

        # ── Interneuronen-Update ──────────────────────────────────────────────
        new_spikes = []   # Spikes dieses Zeitschritts → Motor-Synapsen
        for i in range(N_NEURONS):
            if ref[i] > 0:
                V[i] = V_reset
                ref[i] -= 1
            else:
                dV = (gm[i] * (V_rest - V[i]) + I_ext_arr[i] + I_syn_int[i]) / C_m * dt
                V[i] += dV + noise_int[i] / C_m
                if V[i] >= V_th:
                    V[i]   = V_reset
                    ref[i] = int(tau_ref / dt)
                    spike_rec[i].append(t)
                    s[i]   += 1.0
                    new_spikes.append(i)

        # ── Memristor-Update ──────────────────────────────────────────────────
        I_mem_n      = I_lk * 1e3
        alpha_scaled = alpha * 1e-12 * dt
        dw = alpha_scaled * I_mem_n * biolek_window(w, I_mem_n, p=p_window)
        w  = np.clip(w + dw, 0.0, 1.0)

        # ── Interne synaptische Variablen ─────────────────────────────────────
        s      = np.maximum(s + (-s / tau_syn * dt), 0.0)
        s_buf[:, (step + 1) % buf_len] = s

        # ── Motor-Eingangssynapsen ────────────────────────────────────────────
        # Spike-Liste aus dem Interneuronen-Update verwenden
        for i in new_spikes:
            s_mot[i] += 1.0
        s_mot = np.maximum(s_mot + (-s_mot / tau_syn_mot * dt), 0.0)

        # ── Motor-Synaptischer Strom ──────────────────────────────────────────
        I_syn_mot = np.sum(g_mot_arr * s_mot * (E_syn - V_mot))

        # ── Motorneuron-Update ────────────────────────────────────────────────
        noise_m = rng.normal(0, noise_mot) * np.sqrt(dt)
        if ref_mot > 0:
            V_mot   = V_reset_mot
            ref_mot -= 1
        else:
            g_lk_mot = 1.0 / R_leak_mot
            dV_mot = (g_lk_mot * (V_rest_mot - V_mot) + I_syn_mot) / C_mot * dt
            V_mot  += dV_mot + noise_m / C_mot
            if V_mot >= V_th_mot:
                V_mot   = V_reset_mot
                ref_mot = int(tau_ref_mot / dt)
                spike_mot.append(t)

        # ── Aufzeichnung ──────────────────────────────────────────────────────
        V_rec[:, step]  = V
        w_rec[:, step]  = w
        R_rec[:, step]  = Rm
        V_mot_rec[step] = V_mot

    # ── Motorausgangssignal: Spike-Dichte (geglättet) ─────────────────────────
    t_arr      = np.arange(n_steps) * dt
    motor_out  = np.zeros(n_steps)
    for t_sp in spike_mot:
        idx = int(t_sp / dt)
        if 0 <= idx < n_steps:
            motor_out[idx] += 1.0
    # Gaußsche Glättung (sigma in Zeitschritten)
    sigma_steps = max(1, tau_smooth / dt)
    motor_out   = gaussian_filter1d(motor_out, sigma=sigma_steps)
    # Normierung auf [0, 1]
    mx = motor_out.max()
    if mx > 0:
        motor_out /= mx

    return t_arr, V_rec, w_rec, R_rec, spike_rec, V_mot_rec, spike_mot, motor_out


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⏱ Simulation")
    dt   = st.slider("Zeitschritt dt (ms)", 0.01, 0.5, 0.05, 0.01, format="%.2f")
    T    = st.slider("Gesamtzeit T (ms)", 100, 3000, 800, 50)
    seed = st.number_input("Zufalls-Seed", 0, 9999, 42, 1)

    st.divider()
    st.header("⚡ Interneuronen (Memristor-LIF)")
    C_m     = st.slider("C_m (pF)", 10.0, 500.0, 100.0, 10.0)
    V_rest  = st.slider("V_rest (mV)", -80, -55, -65, 1)
    V_th    = st.slider("V_th (mV)", -55, -40, -50, 1)
    V_reset = st.slider("V_reset (mV)", -85, -60, -70, 1)
    tau_ref = st.slider("τ_ref (ms)", 0.5, 10.0, 2.0, 0.5)

    st.divider()
    st.header("🔴 Memristor")
    R_on     = st.slider("R_on (GΩ)", 0.01, 1.0, 0.1, 0.01, format="%.2f")
    R_off    = st.slider("R_off (GΩ)", 1.0, 50.0, 16.0, 0.5)
    mu_v     = st.slider("μ_v (×10⁻¹⁴ m²/Vs)", 0.1, 20.0, 1.0, 0.1) * 1e-14
    D_nm     = st.slider("D (nm)", 1.0, 30.0, 10.0, 0.5)
    p_window = st.select_slider("Biolek-Exponent p", [1, 2, 3, 4, 5], 1)

    st.divider()
    st.header("🔌 Externer Strom (pA)")
    I_ext_arr = np.array([
        st.slider(f"I_ext N{i+1}", 0.0, 2000.0,
                  [800., 900., 1000., 850., 950.][i], 10.0)
        for i in range(N_NEURONS)
    ])
    noise_sigma = st.slider("Rauschen σ (pA·√ms)", 0.0, 500.0, 50.0, 10.0)

    st.divider()
    st.header("🔗 Interne Kopplung")
    g_syn    = st.slider("g_syn (nS)", 0.0, 5.0, 0.3, 0.05, format="%.2f")
    tau_syn  = st.slider("τ_syn (ms)", 1.0, 30.0, 5.0, 0.5)
    delay_ms = st.slider("Axonale Verzögerung (ms)", 0.1, 10.0, 1.0, 0.1)

    st.divider()
    st.header("🧬 Anfangszustand w₀")
    w0_arr = np.array([
        st.slider(f"w₀ N{i+1}", 0.0, 1.0,
                  [0.3, 0.5, 0.7, 0.4, 0.6][i], 0.05)
        for i in range(N_NEURONS)
    ])

    # ── Motorneuron ───────────────────────────────────────────────────────────
    st.divider()
    st.header("🦾 Motorneuron (Standard-LIF)")
    C_mot        = st.slider("C_mot (pF)", 10.0, 500.0, 150.0, 10.0)
    V_rest_mot   = st.slider("V_rest_mot (mV)", -80, -55, -65, 1)
    V_th_mot     = st.slider("V_th_mot (mV)", -55, -35, -45, 1)
    V_reset_mot  = st.slider("V_reset_mot (mV)", -85, -60, -70, 1)
    tau_ref_mot  = st.slider("τ_ref_mot (ms)", 0.5, 10.0, 2.0, 0.5)
    R_leak_mot   = st.slider("R_leak_mot (GΩ)", 0.1, 20.0, 5.0, 0.1)
    tau_syn_mot  = st.slider("τ_syn_mot (ms)", 1.0, 50.0, 10.0, 1.0)
    noise_mot    = st.slider("Rauschen Motor (pA·√ms)", 0.0, 300.0, 20.0, 5.0)
    tau_smooth   = st.slider("Glättung τ_smooth (ms)", 5.0, 200.0, 40.0, 5.0)

    st.markdown("**Synapsengewichte N_i → Motor (nS)**")
    g_mot_arr = np.array([
        st.slider(f"g_mot N{i+1} (nS)", 0.0, 5.0,
                  [1.0, 1.2, 1.5, 1.0, 1.3][i], 0.05, format="%.2f")
        for i in range(N_NEURONS)
    ])

    run_btn = st.button("▶ Simulation starten", type="primary",
                        use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Modell-Gleichungen
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📐 Physikalisches Modell", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(r"""
**Interneuronen (Memristor-LIF):**

$$C_m \dot{V}_i = g_\mathrm{mem}(w_i)(V_\mathrm{rest}-V_i) + I_\mathrm{ext,i} + I_\mathrm{syn,i} + \sigma\xi_i$$

$$R(w_i) = w_i R_\mathrm{on} + (1-w_i)R_\mathrm{off}$$

$$\dot{w}_i = \alpha I_\mathrm{mem}\cdot[1-(w_i-H(-I))^{2p}]$$

$$\dot{s}_j = -s_j/\tau_\mathrm{syn} + \textstyle\sum_k\delta(t-t_k)$$
""")
    with c2:
        st.markdown("**Motorneuron (Standard-LIF):**")
        st.latex(r"C_\mathrm{mot}\dot{V}_\mathrm{mot} = \frac{V_\mathrm{rest,mot}-V_\mathrm{mot}}{R_\mathrm{lk}} + \sum_i g_{\mathrm{mot},i}\,s_i(t)\,(E_\mathrm{syn}-V_\mathrm{mot}) + \sigma_\mathrm{mot}\xi")
        st.markdown("**Motorausgangssignal (geglättete Feuerrate):**")
        st.latex(r"r(t) = \mathcal{G}_{\tau_s}\!\left[\sum_k \delta(t-t_k^\mathrm{mot})\right] \;\in\; [0,1]")

# ─────────────────────────────────────────────────────────────────────────────
# Simulation ausführen
# ─────────────────────────────────────────────────────────────────────────────
if run_btn or "sim_results" in st.session_state:

    if run_btn:
        delay_steps = max(1, int(delay_ms / dt))
        with st.spinner("Simuliere Netzwerk + Motorneuron…"):
            results = run_simulation(
                dt=dt, T=T,
                C_m=C_m, V_rest=V_rest, V_th=V_th, V_reset=V_reset,
                tau_ref=tau_ref,
                R_on=R_on, R_off=R_off, mu_v=mu_v, D_nm=D_nm,
                p_window=p_window, w0_arr=w0_arr,
                I_ext_arr=I_ext_arr, noise_sigma=noise_sigma,
                g_syn=g_syn, tau_syn=tau_syn, delay_steps=delay_steps,
                C_mot=C_mot, V_rest_mot=V_rest_mot, V_th_mot=V_th_mot,
                V_reset_mot=V_reset_mot, tau_ref_mot=tau_ref_mot,
                R_leak_mot=R_leak_mot,
                g_mot_arr=g_mot_arr, tau_syn_mot=tau_syn_mot,
                noise_mot=noise_mot, tau_smooth=tau_smooth,
                seed=int(seed)
            )
        st.session_state["sim_results"] = results
    else:
        results = st.session_state["sim_results"]

    t_arr, V_rec, w_rec, R_rec, spike_rec, \
        V_mot_rec, spike_mot, motor_out = results

    # ─────────────────────────────────────────────────────────────────────────
    # Kennzahlen
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📊 Spike-Statistik")
    cols = st.columns(N_NEURONS + 1)
    for i in range(N_NEURONS):
        n_sp  = len(spike_rec[i])
        rate  = n_sp / (T * 1e-3)
        isis  = np.diff(spike_rec[i]) if n_sp > 1 else [0]
        cv    = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
        cols[i].metric(f"Neuron {i+1}", f"{n_sp} Spikes",
                       f"{rate:.1f} Hz | CV={cv:.2f}")

    n_mot  = len(spike_mot)
    rate_mot = n_mot / (T * 1e-3)
    peak_out = motor_out.max()
    cols[N_NEURONS].metric(
        "🦾 Motorneuron", f"{n_mot} Spikes",
        f"{rate_mot:.1f} Hz | Ausgang={peak_out:.2f}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Motorausgangssignal – kompakte Anzeige
    # ─────────────────────────────────────────────────────────────────────────
    last_out = float(motor_out[-1])
    st.subheader("🦾 Aktueller Motorbefehl")
    mc1, mc2, mc3 = st.columns([1, 3, 1])
    with mc1:
        st.metric("Motorausgang r(T)", f"{last_out:.3f}")
        st.metric("Feuerrate", f"{rate_mot:.1f} Hz")
    with mc2:
        fig_gauge, ax_g = plt.subplots(figsize=(6, 1.2),
                                        facecolor="#16213e")
        ax_g.set_facecolor("#16213e")
        ax_g.barh(0, last_out, color=COLOR_MOT, height=0.5, alpha=0.9)
        ax_g.barh(0, 1.0, color="#333355", height=0.5, alpha=0.4)
        ax_g.set_xlim(0, 1)
        ax_g.set_ylim(-0.5, 0.5)
        ax_g.set_yticks([])
        ax_g.tick_params(colors="white", labelsize=8)
        for sp in ax_g.spines.values():
            sp.set_color("#444466")
        ax_g.set_xlabel("Motorbefehl r(t)  [normiert]",
                         color="white", fontsize=9)
        ax_g.axvline(last_out, color="white", lw=1.5, ls="--")
        ax_g.text(last_out + 0.02, 0, f"{last_out:.2f}",
                  color="white", va="center", fontsize=9)
        fig_gauge.tight_layout()
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)
    with mc3:
        interpretation = ("STOP" if last_out < 0.2
                          else "LANGSAM" if last_out < 0.5
                          else "MITTEL" if last_out < 0.75
                          else "SCHNELL")
        color_interp = ("#e74c3c" if last_out < 0.2
                        else "#f39c12" if last_out < 0.5
                        else "#2ecc71" if last_out < 0.75
                        else "#00e5ff")
        st.markdown(
            f"<div style='font-size:22px;font-weight:bold;"
            f"color:{color_interp};text-align:center;"
            f"padding:12px;border:2px solid {color_interp};"
            f"border-radius:8px;margin-top:8px'>"
            f"⚡ {interpretation}</div>",
            unsafe_allow_html=True
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Hauptplot: Interneuronen (4 Panels) + Motorneuron (2 Panels)
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📈 Zeitreihen")
    fig = plt.figure(figsize=(14, 15), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(6, 1, hspace=0.5,
                             left=0.07, right=0.97, top=0.95, bottom=0.05)

    ax_V   = fig.add_subplot(gs[0])
    ax_sp  = fig.add_subplot(gs[1], sharex=ax_V)
    ax_w   = fig.add_subplot(gs[2], sharex=ax_V)
    ax_R   = fig.add_subplot(gs[3], sharex=ax_V)
    ax_Vm  = fig.add_subplot(gs[4], sharex=ax_V)
    ax_out = fig.add_subplot(gs[5], sharex=ax_V)

    def style_ax(ax):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_color("#444466")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="major", color="#2a2a5a", linewidth=0.5)
        ax.grid(True, which="minor", color="#1e1e3a", linewidth=0.3)

    for ax in [ax_V, ax_sp, ax_w, ax_R, ax_Vm, ax_out]:
        style_ax(ax)

    # Panel 1 – Interneuronen V(t)
    ax_V.set_title("Interneuronen: Membranpotentiale V(t)", fontsize=10, fontweight="bold")
    ax_V.set_ylabel("V  (mV)", fontsize=9)
    for i in range(N_NEURONS):
        ax_V.plot(t_arr, V_rec[i], color=COLORS[i], lw=0.7, alpha=0.9,
                  label=f"N{i+1}")
    ax_V.axhline(V_th, color="white", lw=0.8, ls="--", alpha=0.4,
                 label=f"V_th={V_th} mV")
    ax_V.legend(fontsize=7, loc="upper right", framealpha=0.2,
                labelcolor="white", facecolor="#1a1a2e")

    # Panel 2 – Spike-Raster (Intern + Motor)
    ax_sp.set_title("Spike-Raster  (N1–N5: Interneuronen · M: Motorneuron)",
                    fontsize=10, fontweight="bold")
    ax_sp.set_ylabel("Einheit", fontsize=9)
    for i in range(N_NEURONS):
        if spike_rec[i]:
            ax_sp.vlines(spike_rec[i], i + 0.55, i + 1.45,
                         color=COLORS[i], lw=1.1, alpha=0.9)
    if spike_mot:
        ax_sp.vlines(spike_mot, N_NEURONS + 0.55, N_NEURONS + 1.45,
                     color=COLOR_MOT, lw=1.4, alpha=0.95)
    ax_sp.set_ylim(0.3, N_NEURONS + 1.7)
    ax_sp.set_yticks(list(range(1, N_NEURONS + 1)) + [N_NEURONS + 1])
    ax_sp.set_yticklabels([f"N{i+1}" for i in range(N_NEURONS)] + ["M"],
                          fontsize=8)
    ax_sp.get_yticklabels()[-1].set_color(COLOR_MOT)

    # Panel 3 – Memristorzustände
    ax_w.set_title("Memristorzustände w_i(t)", fontsize=10, fontweight="bold")
    ax_w.set_ylabel("w  (0…1)", fontsize=9)
    for i in range(N_NEURONS):
        ax_w.plot(t_arr, w_rec[i], color=COLORS[i], lw=0.8, alpha=0.9)
    ax_w.set_ylim(-0.05, 1.05)

    # Panel 4 – Memristiver Widerstand
    ax_R.set_title("Memristiver Widerstand R_i(t)", fontsize=10, fontweight="bold")
    ax_R.set_ylabel("R  (GΩ)", fontsize=9)
    for i in range(N_NEURONS):
        ax_R.plot(t_arr, R_rec[i], color=COLORS[i], lw=0.8, alpha=0.9)

    # Panel 5 – Motorneuron V(t)
    ax_Vm.set_title("Motorneuron: Membranpotential V_mot(t)",
                    fontsize=10, fontweight="bold", color=COLOR_MOT)
    ax_Vm.set_ylabel("V_mot  (mV)", fontsize=9)
    ax_Vm.plot(t_arr, V_mot_rec, color=COLOR_MOT, lw=0.8, alpha=0.95)
    ax_Vm.axhline(V_th_mot, color="white", lw=0.8, ls="--", alpha=0.5,
                  label=f"V_th_mot={V_th_mot} mV")
    ax_Vm.axhline(V_rest_mot, color="#aaaaaa", lw=0.5, ls=":", alpha=0.4)
    if spike_mot:
        ax_Vm.vlines(spike_mot,
                     V_th_mot, V_th_mot + 8,
                     color=COLOR_MOT, lw=0.8, alpha=0.7)
    ax_Vm.legend(fontsize=7, loc="upper right", framealpha=0.2,
                 labelcolor="white", facecolor="#1a1a2e")

    # Panel 6 – Motorausgangssignal
    ax_out.set_title("Motorausgangssignal r(t)  – geglättete Feuerrate [normiert]",
                     fontsize=10, fontweight="bold", color=COLOR_MOT)
    ax_out.set_ylabel("r(t)  [0…1]", fontsize=9)
    ax_out.set_xlabel("Zeit  (ms)", fontsize=9)
    ax_out.fill_between(t_arr, motor_out, alpha=0.35, color=COLOR_MOT)
    ax_out.plot(t_arr, motor_out, color=COLOR_MOT, lw=1.2)
    ax_out.set_ylim(-0.05, 1.1)
    # Interpretationsbänder
    ax_out.axhspan(0.0, 0.2,  alpha=0.08, color="#e74c3c",  label="STOP")
    ax_out.axhspan(0.2, 0.5,  alpha=0.08, color="#f39c12",  label="LANGSAM")
    ax_out.axhspan(0.5, 0.75, alpha=0.08, color="#2ecc71",  label="MITTEL")
    ax_out.axhspan(0.75, 1.1, alpha=0.08, color=COLOR_MOT,  label="SCHNELL")
    ax_out.legend(fontsize=7, loc="upper right", framealpha=0.2,
                  labelcolor="white", facecolor="#1a1a2e")

    fig.suptitle(
        f"Memristor-LIF Netzwerk  ➜  Motorneuron  ·  T={T} ms  ·  dt={dt} ms",
        color="white", fontsize=12, fontweight="bold", y=0.98
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Synapsengewichte (Balkendiagramm)
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("🔗 Synapsengewichte N_i → Motorneuron")
    col_w1, col_w2 = st.columns(2)

    with col_w1:
        fig_w, ax_w2 = plt.subplots(figsize=(5, 3), facecolor="#1a1a2e")
        ax_w2.set_facecolor("#16213e")
        for sp in ax_w2.spines.values():
            sp.set_color("#444466")
        ax_w2.tick_params(colors="white")
        bars = ax_w2.bar(NEURON_LABELS, g_mot_arr, color=COLORS,
                          edgecolor="none", alpha=0.85)
        for bar, gv in zip(bars, g_mot_arr):
            ax_w2.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.03,
                        f"{gv:.2f}", ha="center", va="bottom",
                        color="white", fontsize=9)
        ax_w2.set_ylabel("g_mot (nS)", color="white")
        ax_w2.set_title("Kopplungsstärke → Motorneuron",
                         color="white", fontsize=10)
        fig_w.tight_layout()
        st.pyplot(fig_w, use_container_width=True)
        plt.close(fig_w)

    with col_w2:
        # Feuerrate pro Interneuron vs. Motor-Beitrag
        rates = [len(spike_rec[i]) / (T * 1e-3) for i in range(N_NEURONS)]
        beitrag = [rates[i] * g_mot_arr[i] for i in range(N_NEURONS)]
        fig_b, ax_b = plt.subplots(figsize=(5, 3), facecolor="#1a1a2e")
        ax_b.set_facecolor("#16213e")
        for sp in ax_b.spines.values():
            sp.set_color("#444466")
        ax_b.tick_params(colors="white")
        bars2 = ax_b.bar(NEURON_LABELS, beitrag, color=COLORS,
                          edgecolor="none", alpha=0.85)
        ax_b.set_ylabel("Rate × g_mot  (Hz·nS)", color="white")
        ax_b.set_title("Gewichteter Motorbeitrag pro Neuron",
                        color="white", fontsize=10)
        fig_b.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💾 Daten exportieren")
    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        header = ("t_ms," +
                  ",".join([f"V_N{i+1}_mV,w_N{i+1},R_N{i+1}_GOhm"
                             for i in range(N_NEURONS)]) +
                  ",V_mot_mV,motor_out")
        rows = [header]
        for k in range(len(t_arr)):
            row = f"{t_arr[k]:.3f}"
            for i in range(N_NEURONS):
                row += (f",{V_rec[i,k]:.4f},{w_rec[i,k]:.6f}"
                        f",{R_rec[i,k]:.4f}")
            row += f",{V_mot_rec[k]:.4f},{motor_out[k]:.6f}"
            rows.append(row)
        st.download_button("⬇ Zeitreihen (CSV)", "\n".join(rows),
                           "memristor_lif_motor_timeseries.csv",
                           "text/csv", use_container_width=True)

    with ec2:
        srows = ["source,spike_time_ms"]
        for i in range(N_NEURONS):
            for t_sp in spike_rec[i]:
                srows.append(f"N{i+1},{t_sp:.3f}")
        for t_sp in spike_mot:
            srows.append(f"Motor,{t_sp:.3f}")
        st.download_button("⬇ Spike-Zeiten (CSV)", "\n".join(srows),
                           "memristor_lif_motor_spikes.csv",
                           "text/csv", use_container_width=True)

    with ec3:
        mrows = ["t_ms,motor_out"]
        for k in range(len(t_arr)):
            mrows.append(f"{t_arr[k]:.3f},{motor_out[k]:.6f}")
        st.download_button("⬇ Motorausgangssignal (CSV)", "\n".join(mrows),
                           "motor_output.csv",
                           "text/csv", use_container_width=True)

else:
    st.info("👈  Parameter einstellen und **▶ Simulation starten** drücken.")

# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Referenzen:** Strukov et al. (2008) *Nature* 453:80–83 · "
    "Biolek et al. (2009) *RADIOENGINEERING* 18:210–214 · "
    "FlyWire Connectome (Dorkenwald et al. 2023)  |  "
    "NanoFlyBrain · Prof. Dr. Dietmar Henrich · Medizintechnik"
)

