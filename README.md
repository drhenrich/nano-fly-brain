# Memristor-LIF Neuron Simulator 🧠

Interactive simulation of **5 coupled Memristor-based Leaky Integrate-and-Fire (LIF) neurons**,
implemented as part of the **NanoFlyBrain** project (TU Dresden / Medizintechnik).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## Physical Model

The simulator implements the **HP Memristor model** (Strukov et al., *Nature* 2008)
coupled with classical LIF dynamics:

**LIF membrane potential:**

$$C_m \frac{dV_i}{dt} = g_\text{mem}(w_i)(V_\text{rest} - V_i) + I_{\text{ext},i} + I_{\text{syn},i} + \sigma\xi_i(t)$$

**HP Memristor state (Biolek window function, 2009):**

$$R(w_i) = w_i R_\text{on} + (1-w_i)R_\text{off}, \quad \frac{dw_i}{dt} = \alpha I_\text{mem} \cdot [1-(w-H(-I))^{2p}]$$

**Exponential synaptic coupling with axonal delay:**

$$\dot{s}_j = -s_j/\tau_\text{syn} + \sum_k \delta(t-t_k), \quad I_{\text{syn},i} = g_\text{syn}\sum_{j\neq i} s_j(t-\tau_d)(E_\text{syn}-V_i)$$

## Features

- 5 coupled Memristor-LIF neurons (Euler integration)
- HP Memristor model with Biolek window function (variable exponent *p*)
- Exponential synaptic kinetics with configurable axonal delay
- Gaussian noise (Langevin approach), refractory period
- Interactive sidebar: all physical parameters configurable per neuron
- Visualizations: membrane potentials, spike raster, w(t), R(t), ISI histograms, phase portrait V–w
- CSV export: time series and spike times

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Context

Part of the **NanoFlyBrain** research project:
mapping olfactory *Drosophila* subnetworks (FlyWire connectome)
onto hardware Memristor-LIF arrays for embodied robotics (SNIFFBOT/GoPiGo, ROS).

## References

- Strukov et al. (2008) *Nature* 453:80–83
- Biolek et al. (2009) *RADIOENGINEERING* 18:210–214
- Dorkenwald et al. (2023) FlyWire Connectome, *Nature* 634:124–138

## Author

Prof. Dr. Dietmar Henrich · Physik & Medizintechnik
[profhenrich@googlemail.com](mailto:profhenrich@googlemail.com)
