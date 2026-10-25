"""
app.py - PhysicsGuard Live Demonstration Dashboard & Attack Lab
==============================================================
Streamlit dashboard for live and pre-recorded side-by-side evaluation of
Unprotected vs Plain-Alarm vs PhysicsGuard Protected EV DC fast-charging sessions.

Phase 6.5: Interactive Attack Lab with Composable Attack Builder.

Run via:
    streamlit run app.py
"""

import sys
import os
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Set matplotlib style for dark dashboard aesthetics
plt.style.use("dark_background")
matplotlib_colors = {
    "bg": "#0E1117",
    "card_bg": "#161B22",
    "text": "#E6EDF3",
    "subtext": "#8B949E",
    "border": "#30363D",
    "green": "#2EA043",
    "amber": "#D29922",
    "orange": "#DB6D28",
    "red": "#F85149",
    "cyan": "#388BFD",
    "purple": "#A371F7",
}

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sim_engine import (
    ALL_SCENARIOS, MAIN_DEMO_SCENARIOS, DEMO_CACHE_DIR,
    run_full_simulation, load_from_cache, explain_moment,
    read_live_esp32_sensor, _is_valid_num,
    run_custom_attack_simulation, compute_param_hash,
    load_preset, get_preset_options, PRESET_REGISTRY,
    ALL_TARGETS, ATTACK_TYPES
)
from physicsguard import params

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhysicsGuard | Cyber-Physical EV BMS Defense & Attack Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for Sleek Glassmorphism & High-Contrast Typography
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0E1117;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header badges */
    .badge-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
        flex-wrap: wrap;
    }
    .scenario-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F0F6FC;
        margin: 0;
    }
    .badge-attack {
        background-color: rgba(248, 81, 73, 0.2);
        color: #FF7B72;
        border: 1px solid #F85149;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-fault {
        background-color: rgba(210, 153, 34, 0.2);
        color: #E3B341;
        border: 1px solid #D29922;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-normal {
        background-color: rgba(46, 160, 67, 0.2);
        color: #3FB950;
        border: 1px solid #2EA043;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-mode {
        background-color: rgba(56, 139, 253, 0.15);
        color: #58A6FF;
        border: 1px solid #388BFD;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Explain box callout */
    .explain-box {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95), rgba(13, 17, 23, 0.9));
        border: 1px solid #30363D;
        border-left: 4px solid #58A6FF;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .explain-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #8B949E;
        font-weight: 700;
        margin-bottom: 4px;
        letter-spacing: 0.05em;
    }
    .explain-text {
        font-size: 1.05rem;
        color: #F0F6FC;
        line-height: 1.45;
        font-weight: 400;
    }

    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 10px;
        margin-bottom: 12px;
    }
    .kpi-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 10px 12px;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #8B949E;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .kpi-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #F0F6FC;
    }
    .kpi-danger {
        color: #F85149 !important;
    }
    .kpi-warning {
        color: #D29922 !important;
    }
    .kpi-safe {
        color: #3FB950 !important;
    }

    /* Result Card Styles for Attack Lab */
    .result-card {
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    }
    .result-card-green {
        background: linear-gradient(135deg, rgba(46, 160, 67, 0.16), rgba(22, 27, 34, 0.95));
        border: 2px solid #2EA043;
    }
    .result-card-amber {
        background: linear-gradient(135deg, rgba(210, 153, 34, 0.16), rgba(22, 27, 34, 0.95));
        border: 2px solid #D29922;
    }
    .result-card-red {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.18), rgba(22, 27, 34, 0.95));
        border: 2px solid #F85149;
    }
    .card-header-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 12px;
    }
    .card-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F0F6FC;
    }
    .honesty-notice {
        background: rgba(210, 153, 34, 0.18);
        border: 1px solid #D29922;
        border-radius: 6px;
        padding: 10px 14px;
        color: #F0B72F;
        font-size: 0.92rem;
        font-weight: 500;
        margin-bottom: 14px;
    }

    /* Column headers */
    .col-header-unpr {
        background: rgba(248, 81, 73, 0.12);
        border: 1px solid rgba(248, 81, 73, 0.3);
        border-radius: 6px;
        padding: 8px 12px;
        text-align: center;
        font-weight: 700;
        color: #FF7B72;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
    .col-header-prot {
        background: rgba(46, 160, 67, 0.12);
        border: 1px solid rgba(46, 160, 67, 0.3);
        border-radius: 6px;
        padding: 8px 12px;
        text-align: center;
        font-weight: 700;
        color: #3FB950;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
    .col-header-baseline {
        background: rgba(210, 153, 34, 0.12);
        border: 1px solid rgba(210, 153, 34, 0.3);
        border-radius: 6px;
        padding: 8px 12px;
        text-align: center;
        font-weight: 700;
        color: #E3B341;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }

    /* Subtitle tag */
    .gt-tag {
        font-size: 0.78rem;
        color: #8B949E;
        font-style: italic;
        margin-top: -6px;
        margin-bottom: 6px;
        display: block;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable Plot Rendering Functions
# ---------------------------------------------------------------------------
def render_side_by_side_plots(df_u: pd.DataFrame, df_p: pd.DataFrame, curr_t: float):
    """Render Unprotected vs Protected trajectory comparison plots side by side."""
    col_plot_u, col_plot_p = st.columns(2)

    with col_plot_u:
        st.markdown("#### Unprotected Trajectory")
        st.markdown('<span class="gt-tag">Ground truth physical plant state unobserved by the cyber controller.</span>', unsafe_allow_html=True)
        
        fig_u, axes_u = plt.subplots(3, 1, figsize=(7.2, 7.8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.2, 1.0]})
        fig_u.patch.set_facecolor(matplotlib_colors["card_bg"])
        for ax in axes_u:
            ax.set_facecolor(matplotlib_colors["bg"])
            ax.grid(True, linestyle="--", alpha=0.2, color="#8B949E")
            ax.axvline(curr_t, color=matplotlib_colors["cyan"], linestyle=":", linewidth=1.5, alpha=0.8)

        # 1. Ground Truth Battery Temperature
        ax1 = axes_u[0]
        ax1.set_title("Ground Truth Battery Temperature", fontsize=10, fontweight="bold", color="#FF7B72", pad=6)
        ax1.axhspan(45.0, 60.0, color=matplotlib_colors["red"], alpha=0.15, label="Danger Zone (>45°C)")
        ax1.axhline(45.0, color=matplotlib_colors["amber"], linestyle="--", linewidth=1.2, label="45°C Derating Limit")
        ax1.axhline(50.0, color=matplotlib_colors["red"], linestyle="--", linewidth=1.2, label="50°C Thermal Trip")
        ax1.plot(df_u["time_s"], df_u["true_battery_temp"], color="#FF7B72", linewidth=2.0, label="True Battery Temp")
        ax1.set_ylabel("Temp (°C)", fontsize=9)
        ax1.set_ylim(20.0, 55.0)
        ax1.legend(loc="upper left", fontsize=8, framealpha=0.4)

        # 2. Measured Sensors
        ax2 = axes_u[1]
        ax2.set_title("Measured Cell Temperatures (What Controller Sees)", fontsize=10, fontweight="bold", color=matplotlib_colors["text"], pad=6)
        ax2.plot(df_u["time_s"], df_u["temp_cell_a"], color="#58A6FF", linewidth=1.5, label="Sensor A (temp_cell_a)")
        ax2.plot(df_u["time_s"], df_u["temp_cell_b"], color="#A371F7", linewidth=1.5, linestyle="--", label="Sensor B (temp_cell_b)")
        ax2.set_ylabel("Temp (°C)", fontsize=9)
        ax2.set_ylim(20.0, 55.0)
        ax2.legend(loc="upper left", fontsize=8, framealpha=0.4)

        # 3. Actuator Commands
        ax3 = axes_u[2]
        ax3.set_title("Actuator Delivery (Current & Pump Flow)", fontsize=10, fontweight="bold", color=matplotlib_colors["text"], pad=6)
        l1 = ax3.plot(df_u["time_s"], df_u["applied_current_a"], color="#2EA043", linewidth=1.8, label="Applied Current (A)")
        ax3.set_ylabel("Current (A)", color="#2EA043", fontsize=9)
        ax3.set_ylim(-5, 260)
        ax3.set_xlabel("Time (s)", fontsize=9)

        ax3_twin = ax3.twinx()
        l2 = ax3_twin.plot(df_u["time_s"], df_u["applied_pump_lpm"], color="#388BFD", linewidth=1.6, linestyle="-.", label="Pump Flow (L/min)")
        ax3_twin.set_ylabel("Flow (L/min)", color="#388BFD", fontsize=9)
        ax3_twin.set_ylim(-1, 22)

        lines_u = l1 + l2
        labels_u = [l.get_label() for l in lines_u]
        ax3.legend(lines_u, labels_u, loc="upper right", fontsize=8, framealpha=0.4)

        plt.tight_layout()
        st.pyplot(fig_u)
        plt.close(fig_u)

    with col_plot_p:
        st.markdown("#### PhysicsGuard Protected Trajectory")
        st.markdown('<span class="gt-tag">Actuator Gate Screening + Digital Twin Virtual Sensor on SIL-3 Hardware Boundary.</span>', unsafe_allow_html=True)
        
        fig_p, axes_p = plt.subplots(4, 1, figsize=(7.2, 8.8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.2, 1.0, 0.45]})
        fig_p.patch.set_facecolor(matplotlib_colors["card_bg"])
        for ax in axes_p:
            ax.set_facecolor(matplotlib_colors["bg"])
            ax.grid(True, linestyle="--", alpha=0.2, color="#8B949E")
            ax.axvline(curr_t, color=matplotlib_colors["cyan"], linestyle=":", linewidth=1.5, alpha=0.8)

        # 1. Protected True Temperature
        ax1_p = axes_p[0]
        ax1_p.set_title("Protected True Battery Temperature", fontsize=10, fontweight="bold", color="#3FB950", pad=6)
        ax1_p.axhspan(45.0, 60.0, color=matplotlib_colors["red"], alpha=0.15, label="Danger Zone (>45°C)")
        ax1_p.axhline(45.0, color=matplotlib_colors["amber"], linestyle="--", linewidth=1.2, label="45°C Derating Limit")
        ax1_p.axhline(50.0, color=matplotlib_colors["red"], linestyle="--", linewidth=1.2, label="50°C Thermal Trip")
        ax1_p.plot(df_p["time_s"], df_p["true_battery_temp"], color="#3FB950", linewidth=2.0, label="Protected True Temp")
        ax1_p.set_ylabel("Temp (°C)", fontsize=9)
        ax1_p.set_ylim(20.0, 55.0)
        ax1_p.legend(loc="upper left", fontsize=8, framealpha=0.4)

        # 2. Sensors A & B vs Trusted Virtual Temperature
        ax2_p = axes_p[1]
        ax2_p.set_title("Sensors A & B vs Gate-Trusted Temperature", fontsize=10, fontweight="bold", color=matplotlib_colors["text"], pad=6)
        ax2_p.plot(df_p["time_s"], df_p["temp_cell_a"], color="#58A6FF", linewidth=1.4, alpha=0.7, label="Sensor A (raw)")
        ax2_p.plot(df_p["time_s"], df_p["temp_cell_b"], color="#A371F7", linewidth=1.4, linestyle="--", alpha=0.7, label="Sensor B (raw)")
        ax2_p.plot(df_p["time_s"], df_p["trusted_temp"], color="#3FB950", linewidth=2.0, label="Trusted Temp (Gate Output)")
        ax2_p.set_ylabel("Temp (°C)", fontsize=9)
        ax2_p.set_ylim(20.0, 55.0)
        ax2_p.legend(loc="upper left", fontsize=8, framealpha=0.4)

        # 3. Protected Actuation
        ax3_p = axes_p[2]
        ax3_p.set_title("Safe Actuation (Clamped Current & Interlocked Pump)", fontsize=10, fontweight="bold", color=matplotlib_colors["text"], pad=6)
        lp1 = ax3_p.plot(df_p["time_s"], df_p["applied_current_a"], color="#2EA043", linewidth=1.8, label="Delivered Current (A)")
        ax3_p.set_ylabel("Current (A)", color="#2EA043", fontsize=9)
        ax3_p.set_ylim(-5, 260)

        ax3_p_twin = ax3_p.twinx()
        lp2 = ax3_p_twin.plot(df_p["time_s"], df_p["applied_pump_lpm"], color="#388BFD", linewidth=1.6, linestyle="-.", label="Delivered Pump (L/min)")
        ax3_p_twin.set_ylabel("Flow (L/min)", color="#388BFD", fontsize=9)
        ax3_p_twin.set_ylim(-1, 22)

        lines_p = lp1 + lp2
        labels_p = [l.get_label() for l in lines_p]
        ax3_p.legend(lines_p, labels_p, loc="upper right", fontsize=8, framealpha=0.4)

        # 4. Gate Operating Mode Band
        ax4_p = axes_p[3]
        ax4_p.set_title("Gate Operating Mode Band", fontsize=9, fontweight="bold", color=matplotlib_colors["text"], pad=4)
        mode_map = {"NOMINAL": 0, "DEGRADED": 1, "LIMP_HOME": 2, "SHUTDOWN": 3}
        mode_colors = ["#2EA043", "#D29922", "#DB6D28", "#F85149"]
        
        modes_num = df_p["gate_mode"].map(lambda m: mode_map.get(m, 0)).values
        t_vals = df_p["time_s"].values

        for i in range(len(t_vals) - 1):
            m_idx = modes_num[i]
            ax4_p.axvspan(t_vals[i], t_vals[i+1], color=mode_colors[m_idx], alpha=0.85)

        ax4_p.set_yticks([0.5])
        ax4_p.set_yticklabels(["Mode"], fontsize=8)
        ax4_p.set_xlabel("Time (s)", fontsize=9)

        legend_patches = [
            mpatches.Patch(color=mode_colors[0], label="NOMINAL"),
            mpatches.Patch(color=mode_colors[1], label="DEGRADED"),
            mpatches.Patch(color=mode_colors[2], label="LIMP_HOME"),
            mpatches.Patch(color=mode_colors[3], label="SHUTDOWN")
        ]
        ax4_p.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.0, 1.4), ncol=4, fontsize=7, framealpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)


def render_residual_small_multiples(df_p: pd.DataFrame, curr_t: float, thr: Dict[str, float]):
    """Render 4x2 grid of physics residual observers and diagnosis timeline."""
    st.markdown("### 🔬 Physics Residual Observers & Multi-Channel Isolation")
    st.caption("Each residual evaluates a distinct physical law (thermal conservation, 1-RC electrical tracking, coolant energy balance, hydraulic continuity).")

    r_specs = [
        ("R1a", "R1a: Cell A vs Twin (°C)", thr.get("R1_threshold_c", 5.04), [-2, 25], "#58A6FF"),
        ("R1b", "R1b: Cell B vs Twin (°C)", thr.get("R1_threshold_c", 5.04), [-2, 25], "#A371F7"),
        ("R2", "R2: Coolant Balance (kJ)", thr.get("R2_threshold_kj", 4.91), [-1, 15], "#3FB950"),
        ("R3", "R3: Redundancy |A - B| (°C)", thr.get("R3_threshold_c", 1.44), [-1, 25], "#D29922"),
        ("Rv", "Rv: Pack Voltage Residual (V)", thr.get("Rv_threshold_v", 10.61), [-5, 50], "#F0883E"),
        ("Rflow", "Rflow: Pump Hydraulic (L/min)", thr.get("Rflow_threshold", 0.55), [-1, 8], "#79C0FF"),
        ("Rcur", "Rcur: Pack Current Residual (A)", thr.get("Rcur_threshold", 10.0), [-2, 50], "#FF7B72"),
    ]

    fig_res, axes_res = plt.subplots(4, 2, figsize=(14, 8), sharex=True)
    axes_flat = axes_res.flatten()
    fig_res.patch.set_facecolor(matplotlib_colors["card_bg"])

    for idx, (col_name, title, limit, y_limits, col_color) in enumerate(r_specs):
        ax = axes_flat[idx]
        ax.set_facecolor(matplotlib_colors["bg"])
        ax.grid(True, linestyle="--", alpha=0.2, color="#8B949E")
        ax.axvline(curr_t, color=matplotlib_colors["cyan"], linestyle=":", linewidth=1.5, alpha=0.8)

        if col_name in df_p.columns and df_p[col_name].notna().any():
            ax.plot(df_p["time_s"], df_p[col_name].abs(), color=col_color, linewidth=1.5, label="|Residual|")
        
        ax.axhline(limit, color=matplotlib_colors["red"], linestyle="--", linewidth=1.2, label=f"Thr ({limit:.2f})")
        ax.set_title(title, fontsize=9, fontweight="bold", color=matplotlib_colors["text"], pad=4)
        ax.set_ylim(y_limits)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.3)

    # 8th subplot: Diagnosis Class Timeline
    ax_diag = axes_flat[7]
    ax_diag.set_facecolor(matplotlib_colors["bg"])
    ax_diag.grid(True, linestyle="--", alpha=0.2, color="#8B949E")
    ax_diag.axvline(curr_t, color=matplotlib_colors["cyan"], linestyle=":", linewidth=1.5, alpha=0.8)
    ax_diag.set_title("Guard Diagnosis Category", fontsize=9, fontweight="bold", color=matplotlib_colors["text"], pad=4)

    diag_map = {
        "NORMAL": 0,
        "SENSOR_HW_FAULT": 1,
        "SENSOR_FROZEN": 2,
        "SENSOR_SPOOF": 3,
        "ACTUATOR_FAULT": 4,
        "COMMAND_RESPONSE_MISMATCH": 5,
        "UNSAFE_COMMAND": 6,
    }
    diag_colors = ["#2EA043", "#F0883E", "#A371F7", "#F85149", "#D29922", "#388BFD", "#FF7B72"]

    if "diagnosis" in df_p.columns:
        d_nums = df_p["diagnosis"].map(lambda d: diag_map.get(d, 0)).values
        t_d = df_p["time_s"].values
        for i in range(len(t_d) - 1):
            ax_diag.axvspan(t_d[i], t_d[i+1], color=diag_colors[d_nums[i]], alpha=0.85)

    ax_diag.set_yticks([0.5])
    ax_diag.set_yticklabels(["Class"], fontsize=8)
    ax_diag.set_xlabel("Time (s)", fontsize=9)

    diag_legend_handles = [
        mpatches.Patch(color="#2EA043", label="NORMAL"),
        mpatches.Patch(color="#A371F7", label="FROZEN"),
        mpatches.Patch(color="#F85149", label="SPOOF"),
        mpatches.Patch(color="#D29922", label="ACTUATOR"),
        mpatches.Patch(color="#388BFD", label="MISMATCH"),
    ]
    ax_diag.legend(handles=diag_legend_handles, loc="upper right", bbox_to_anchor=(1.0, 1.4), ncol=5, fontsize=7, framealpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_res)
    plt.close(fig_res)


# Load thresholds for small multiples
thr_file = ROOT_DIR / "outputs" / "thresholds.json"
thr = {}
if thr_file.exists():
    try:
        with open(thr_file) as f:
            thr = json.load(f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cached Simulation Runner for Main Scenario Explorer
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_simulation_data(
    scenario_name: str,
    ambient_c: Optional[float],
    inlet_c: Optional[float],
    seed: int,
    start_s: Optional[float],
    use_cache: bool
) -> Dict[str, Any]:
    """Retrieve simulation data either from pre-recorded cache or live run."""
    if use_cache:
        data = load_from_cache(scenario_name)
        if data is not None:
            return data
    return run_full_simulation(
        scenario_name, ambient_c=ambient_c, inlet_c=inlet_c,
        seed=seed, start_s=start_s
    )


# ---------------------------------------------------------------------------
# Sidebar Navigation & Controls
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://img.shields.io/badge/PhysicsGuard-Phase%206.5%20Attack%20Lab-388bfd?style=for-the-badge&logo=shield",
    use_container_width=True
)
st.sidebar.markdown("### 🎛️ Scenario Explorer Controls")

scenario_selection = st.sidebar.selectbox(
    "Select Scenario",
    options=ALL_SCENARIOS,
    index=0,
    format_func=lambda s: f"🔥 {s.upper()}" if s in MAIN_DEMO_SCENARIOS else f"⚡ {s}"
)

is_main_demo = scenario_selection in MAIN_DEMO_SCENARIOS
default_amb = 35.0 if is_main_demo else 25.0
default_inlet = 35.0 if is_main_demo else 25.0

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    amb_temp = st.slider("Ambient Air (°C)", min_value=15.0, max_value=45.0, value=default_amb, step=1.0)
with col_sb2:
    inlet_temp = st.slider("Coolant Inlet (°C)", min_value=15.0, max_value=45.0, value=default_inlet, step=1.0)

col_sb3, col_sb4 = st.sidebar.columns(2)
with col_sb3:
    seed_val = st.number_input("RNG Seed", min_value=0, max_value=9999, value=42, step=1)
with col_sb4:
    start_s_val = st.number_input("Attack Start (s)", min_value=0.0, max_value=1200.0, value=300.0, step=50.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Demo Mode")
use_saved_data = st.sidebar.toggle("Use saved demo data (Seed 42)", value=True, help="Replays instant pre-recorded CSVs from demo_cache/ in Scenario Explorer")
live_sensor_toggle = st.sidebar.toggle("Live hardware sensor (ESP32 / Serial)", value=False, help="Polls serial port for external temperature; falls back to simulation silently")
run_clicked = st.sidebar.button("🚀 Re-Run Explorer Simulation", use_container_width=True)

live_temp_reading = None
if live_sensor_toggle:
    live_temp_reading = read_live_esp32_sensor()
    if live_temp_reading is not None:
        st.sidebar.success(f"Hardware sensor connected: {live_temp_reading:.1f} °C")
    else:
        st.sidebar.info("No active serial sensor detected. Operating on high-fidelity simulation.")


# ---------------------------------------------------------------------------
# Navigation Tabs (Attack Lab prominent as First Tab)
# ---------------------------------------------------------------------------
tab_lab, tab_demo, tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Attack Lab (Phase 6.5)",
    "🚗 Scenario Explorer (Phases 1-6)",
    "📐 How It Works",
    "📊 Physics Residual Matrix",
    "📈 Evaluation Results & Held-Out Suite",
    "📋 Limitations & Technical Assumptions"
])


# ===========================================================================
# TAB 1: ATTACK LAB (PHASE 6.5)
# ===========================================================================
with tab_lab:
    st.markdown("## 🧪 Cyberattack Lab: Live Adversarial Evaluation")
    st.caption("Craft or choose a custom attack against the EV plant & BMS controller. PhysicsGuard detects and SafetyGate mitigates side-by-side with no protection and a plain 45 °C alarm baseline.")

    # Initialize Attack Lab Session State
    if "lab_specs" not in st.session_state:
        init_specs, _ = load_preset("demo_hot_attack")
        st.session_state.lab_specs = init_specs
    if "lab_amb" not in st.session_state:
        st.session_state.lab_amb = 35.0
    if "lab_inlet" not in st.session_state:
        st.session_state.lab_inlet = 35.0
    if "lab_seed" not in st.session_state:
        st.session_state.lab_seed = 42
    if "lab_variation" not in st.session_state:
        st.session_state.lab_variation = False
    if "lab_playback_time" not in st.session_state:
        st.session_state.lab_playback_time = 120

    # 1. Preset & Environment Controls
    st.markdown("### 🎛️ Attack Builder & Physical Environment")
    col_preset, col_add = st.columns([3, 1])

    with col_preset:
        preset_choices = ["(Custom Attack)"] + get_preset_options()
        selected_preset = st.selectbox(
            "Load Attack Recipe Preset (1-Click Demo)",
            options=preset_choices,
            index=1,  # Default to demo_hot_attack or pump_off
            help="Select a pre-configured attack vector to immediately populate the builder."
        )

        # Handle preset selection change
        if "last_selected_preset" not in st.session_state:
            st.session_state.last_selected_preset = selected_preset

        if selected_preset != st.session_state.last_selected_preset and selected_preset != "(Custom Attack)":
            p_specs, p_env = load_preset(selected_preset)
            st.session_state.lab_specs = p_specs
            if "ambient_c" in p_env:
                st.session_state.lab_amb = float(p_env["ambient_c"])
            if "inlet_c" in p_env:
                st.session_state.lab_inlet = float(p_env["inlet_c"])
            st.session_state.last_selected_preset = selected_preset
            st.rerun()

    col_env1, col_env2, col_env3, col_env4 = st.columns(4)
    with col_env1:
        st.session_state.lab_amb = st.slider("Ambient Air (°C)", 15.0, 45.0, value=float(st.session_state.lab_amb), step=1.0, key="slider_lab_amb")
    with col_env2:
        st.session_state.lab_inlet = st.slider("Coolant Inlet (°C)", 15.0, 45.0, value=float(st.session_state.lab_inlet), step=1.0, key="slider_lab_inlet")
    with col_env3:
        st.session_state.lab_seed = st.number_input("RNG Seed", min_value=0, max_value=9999, value=int(st.session_state.lab_seed), step=1, key="num_lab_seed")
    with col_env4:
        st.session_state.lab_variation = st.toggle("Plant variation (±10%)", value=bool(st.session_state.lab_variation), key="tog_lab_variation", help="Perturbs physical plant parameters (resistance, heat capacity, pump lag) by ±10% from nominal twin")

    # Allowed targets and types as restricted by Scope Cut
    ALLOWED_TARGETS = ["temp_cell_a", "temp_cell_b", "pump_flow_lpm", "cmd_current", "cmd_pump", "coolant_temp_out"]
    ALLOWED_TYPES = ["freeze", "drift", "spike", "set_value", "dead_nan"]

    # 2. Attack Builder Rows (Up to 2 stacked attacks)
    st.markdown("##### Attack Specification Rows (Stacked Layers: Max 2)")
    
    current_specs = list(st.session_state.lab_specs)
    if len(current_specs) == 0:
        current_specs = [{"target": "cmd_pump", "type": "set_value", "magnitude": 0.0, "start_s": 300.0, "duration_s": None}]

    updated_specs = []
    for idx, spec in enumerate(current_specs):
        with st.container():
            st.markdown(f"**Layer {idx + 1}**")
            c_tgt, c_type, c_mag, c_start, c_dur = st.columns([1.5, 1.3, 1.2, 1.2, 1.2])

            with c_tgt:
                tgt_val = spec.get("target", "temp_cell_a")
                tgt_idx = ALLOWED_TARGETS.index(tgt_val) if tgt_val in ALLOWED_TARGETS else 0
                tgt = st.selectbox(f"Target", options=ALLOWED_TARGETS, index=tgt_idx, key=f"lab_tgt_{idx}")

            with c_type:
                type_val = spec.get("type", "freeze")
                type_idx = ALLOWED_TYPES.index(type_val) if type_val in ALLOWED_TYPES else 0
                atk_type = st.selectbox(f"Type", options=ALLOWED_TYPES, index=type_idx, key=f"lab_type_{idx}")

            with c_mag:
                # Default/step guidance based on target and type
                mag_default = float(spec.get("magnitude", 0.0))
                mag = st.number_input(f"Magnitude / Rate", value=mag_default, step=1.0, key=f"lab_mag_{idx}", help="For drift: °C/min or A/min. For set_value: constant value. For spike: step offset.")

            with c_start:
                start_val = float(spec.get("start_s", 300.0))
                start_s = st.number_input(f"Start (s)", min_value=0.0, max_value=3000.0, value=start_val, step=50.0, key=f"lab_start_{idx}")

            with c_dur:
                dur_raw = spec.get("duration_s")
                dur_val = float(dur_raw) if dur_raw is not None and str(dur_raw).strip() != "" and float(dur_raw) > 0 else 0.0
                dur_s = st.number_input(f"Duration s (0 = End)", min_value=0.0, max_value=7200.0, value=dur_val, step=30.0, key=f"lab_dur_{idx}")
                eff_dur = None if dur_s <= 0.0 else dur_s

            updated_specs.append({
                "target": tgt,
                "type": atk_type,
                "magnitude": mag,
                "start_s": start_s,
                "duration_s": eff_dur
            })

    st.session_state.lab_specs = updated_specs

    # Add/Remove Layer Buttons
    c_btn1, c_btn2, _ = st.columns([1, 1, 3])
    with c_btn1:
        if st.button("➕ Add Attack Layer", disabled=len(st.session_state.lab_specs) >= 2):
            st.session_state.lab_specs.append({
                "target": "cmd_pump", "type": "set_value", "magnitude": 0.0, "start_s": 300.0, "duration_s": None
            })
            st.rerun()
    with c_btn2:
        if st.button("➖ Remove Layer", disabled=len(st.session_state.lab_specs) <= 1):
            st.session_state.lab_specs.pop()
            st.rerun()

    # 3. Execution Action
    col_run_lab, col_cache_hash = st.columns([2, 3])
    with col_run_lab:
        run_lab_btn = st.button("⚡ Run Attack Lab Simulation", type="primary", use_container_width=True)

    curr_p_hash = compute_param_hash(
        st.session_state.lab_specs,
        ambient_c=st.session_state.lab_amb,
        inlet_c=st.session_state.lab_inlet,
        seed=st.session_state.lab_seed,
        plant_variation=st.session_state.lab_variation
    )
    with col_cache_hash:
        st.caption(f"Parameter Signature: `{curr_p_hash}` (Sub-second execution via deterministic parameter hash cache).")

    # Run or retrieve attack simulation
    if "lab_res" not in st.session_state or run_lab_btn:
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def _lab_progress(frac: float, msg: str):
            progress_bar.progress(frac)
            status_box.caption(f"⚙️ {msg}")

        lab_res = run_custom_attack_simulation(
            attack_specs=st.session_state.lab_specs,
            ambient_c=st.session_state.lab_amb,
            inlet_c=st.session_state.lab_inlet,
            seed=st.session_state.lab_seed,
            plant_variation=st.session_state.lab_variation,
            timeout_wall_clock_s=30.0,
            progress_callback=_lab_progress
        )

        progress_bar.empty()
        status_box.empty()
        st.session_state.lab_res = lab_res

    lab_res = st.session_state.lab_res

    # Check for execution error or timeout
    if "error" in lab_res:
        st.error(f"⚠️ {lab_res['error']}")
    else:
        lab_kpis = lab_res["kpis"]
        lab_df_u = lab_res["df_unpr"]
        lab_df_p = lab_res["df_prot"]
        lab_df_plain = lab_res["df_plain"]
        lab_events = lab_res.get("gate_events", [])
        lab_verdicts = lab_res.get("verdicts", [])
        lab_timeline = lab_res.get("timeline", [])

        # -------------------------------------------------------------------
        # RESULT CARD (Honesty Rules & KPI Grid)
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📋 Attack Lab Results")

        card_status = lab_kpis.get("card_status", "amber")
        card_class = f"result-card result-card-{card_status}"
        detected_text = lab_kpis.get("detected_label", "NOT DETECTED")
        is_det = (detected_text == "YES")
        det_badge_class = "badge-attack" if is_det else "badge-fault"
        lat_str = f"+{lab_kpis['detection_latency_s']:.1f} s latency" if is_det and lab_kpis.get("detection_latency_s") is not None else "No detection"

        blamed_str = ", ".join(lab_kpis.get("blamed_signals", [])) if lab_kpis.get("blamed_signals") else "None"
        diagnoses_str = ", ".join(lab_kpis.get("diagnoses_seen", [])) if lab_kpis.get("diagnoses_seen") else "NORMAL"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="card-header-flex">
                <span class="card-title">ATTACK LAB EVALUATION SUMMARY</span>
                <span class="{det_badge_class}">GUARD STATUS: {detected_text} ({lat_str})</span>
                <span class="badge-mode">DIAGNOSIS: {diagnoses_str}</span>
                <span class="badge-mode">BLAMED: {blamed_str}</span>
                <span class="badge-mode">FINAL GATE: {lab_kpis.get('final_mode', 'NOMINAL')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Honesty Rule: Call out when attack produces no physical effect in this model
        if not lab_kpis.get("has_physical_effect", True):
            st.markdown(f"""
            <div class="honesty-notice">
                ⚠️ <strong>HONESTY RULE NOTICE — {lab_kpis['physical_effect_label']}:</strong> 
                In this mathematical model, the fast-charging BMS controller does not close loop on the attacked signal channel.
                Consequently, true cell temperatures, applied charging current, and battery SOC remain strictly identical to the healthy normal baseline run.
            </div>
            """, unsafe_allow_html=True)

        # Three-column KPI comparison (Unprotected vs Plain Alarm vs Protected)
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)

        with c_kpi1:
            st.markdown('<div class="col-header-unpr">UNPROTECTED (NO DEFENSE)</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Peak Temp</div>
                    <div class="kpi-value {'kpi-danger' if lab_kpis['peak_t_unpr'] > 45.0 else 'kpi-safe'}">{lab_kpis['peak_t_unpr']:.1f} °C</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Time &gt;45°C</div>
                    <div class="kpi-value {'kpi-danger' if lab_kpis['s45_unpr'] > 0 else 'kpi-safe'}">{lab_kpis['s45_unpr']} s</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Time &gt;50°C</div>
                    <div class="kpi-value {'kpi-danger' if lab_kpis['s50_unpr'] > 0 else 'kpi-safe'}">{lab_kpis['s50_unpr']} s</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Charge Completed?</div>
                    <div class="kpi-value {'kpi-safe' if lab_kpis['completed_unpr'] else 'kpi-danger'}">{'YES' if lab_kpis['completed_unpr'] else 'NO'} ({lab_kpis['soc_unpr']*100:.1f}%)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Session Time</div>
                    <div class="kpi-value">{lab_kpis['dur_unpr']:.0f} s</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_kpi2:
            st.markdown('<div class="col-header-baseline">PLAIN-ALARM (45°C TRIP)</div>', unsafe_allow_html=True)
            plain_status_str = "TRIPPED" if lab_kpis["plain_tripped"] else "NOT TRIPPED"
            plain_cls = "kpi-warning" if lab_kpis["plain_tripped"] else ("kpi-danger" if lab_kpis["peak_t_plain"] > 45.0 else "kpi-safe")
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Trip Status</div>
                    <div class="kpi-value {plain_cls}">{plain_status_str}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Peak Temp</div>
                    <div class="kpi-value {'kpi-danger' if lab_kpis['peak_t_plain'] > 45.0 else 'kpi-safe'}">{lab_kpis['peak_t_plain']:.1f} °C</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Time &gt;45°C</div>
                    <div class="kpi-value {'kpi-danger' if lab_kpis['s45_plain'] > 0 else 'kpi-safe'}">{lab_kpis['s45_plain']} s</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Charge Completed?</div>
                    <div class="kpi-value {'kpi-safe' if lab_kpis['soc_plain'] >= 0.795 else 'kpi-warning'}">{'YES' if lab_kpis['soc_plain'] >= 0.795 else 'TRIPPED'} ({lab_kpis['soc_plain']*100:.1f}%)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Session Time</div>
                    <div class="kpi-value">{lab_kpis['dur_plain']:.0f} s</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_kpi3:
            st.markdown('<div class="col-header-prot">PHYSICSGUARD PROTECTED</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Peak Temp</div>
                    <div class="kpi-value {'kpi-safe' if lab_kpis['peak_t_prot'] <= 45.0 else 'kpi-danger'}">{lab_kpis['peak_t_prot']:.1f} °C</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Time &gt;45°C</div>
                    <div class="kpi-value kpi-safe">{lab_kpis['s45_prot']} s</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Time &gt;50°C</div>
                    <div class="kpi-value kpi-safe">{lab_kpis['s50_prot']} s</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Charge Completed?</div>
                    <div class="kpi-value {'kpi-safe' if lab_kpis['completed_prot'] else 'kpi-danger'}">{'YES' if lab_kpis['completed_prot'] else 'NO'} ({lab_kpis['soc_prot']*100:.1f}%)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Duration vs Base</div>
                    <div class="kpi-value">{lab_kpis['dur_prot']:.0f} s ({lab_kpis['penalty_pct']:+3.1f}%)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # "WHAT HAPPENED" TIMELINE BOX
        # -------------------------------------------------------------------
        st.markdown("#### 💬 What Happened: Plain-English Physical Timeline")
        timeline_html = "<div class='explain-box'><div class='explain-title'>Chronological Incident Reconstruction</div>"
        for item in lab_timeline:
            timeline_html += f"<div class='explain-text' style='margin-bottom: 6px;'>• {item}</div>"
        timeline_html += "</div>"
        st.markdown(timeline_html, unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # ANIMATED REPLAY & TIME SCRUBBER
        # -------------------------------------------------------------------
        lab_max_time = int(max(
            float(lab_df_u["time_s"].max()) if len(lab_df_u) > 0 else 0,
            float(lab_df_p["time_s"].max()) if len(lab_df_p) > 0 else 0
        ))

        st.markdown("#### ⏱️ Animated Replay & Playback Timeline")
        col_r1, col_r2, col_r3, col_r4 = st.columns([5, 1, 1, 1])
        with col_r1:
            scrub_lab = st.slider(
                "Scrub Timeline (s)",
                min_value=1,
                max_value=lab_max_time,
                value=min(st.session_state.lab_playback_time, lab_max_time),
                step=1,
                key="lab_timeline_slider"
            )
            st.session_state.lab_playback_time = scrub_lab

        first_atk_start = int(st.session_state.lab_specs[0].get("start_s", 300)) if st.session_state.lab_specs else 300

        with col_r2:
            if st.button("⏮ Start", key="lab_btn_start"):
                st.session_state.lab_playback_time = 1
                st.rerun()
        with col_r3:
            if st.button(f"⚡ Attack ({first_atk_start}s)", key="lab_btn_atk"):
                st.session_state.lab_playback_time = min(first_atk_start, lab_max_time)
                st.rerun()
        with col_r4:
            if st.button("⏭ End", key="lab_btn_end"):
                st.session_state.lab_playback_time = lab_max_time
                st.rerun()

        curr_lab_t = st.session_state.lab_playback_time

        # Explain This Moment Box for Attack Lab
        idx_lab_p = (lab_df_p["time_s"] - curr_lab_t).abs().argmin() if len(lab_df_p) > 0 else 0
        row_lab_p = lab_df_p.iloc[idx_lab_p] if len(lab_df_p) > 0 else pd.Series()
        verdict_lab_curr = lab_verdicts[idx_lab_p] if lab_verdicts and idx_lab_p < len(lab_verdicts) else None

        recent_lab_ev = None
        for ev in reversed(lab_events):
            if ev.get("t", 0) <= curr_lab_t:
                recent_lab_ev = ev
                break

        moment_lab_text = explain_moment(curr_lab_t, row_lab_p, verdict_lab_curr, recent_lab_ev)
        st.markdown(f"""
        <div class="explain-box">
            <div class="explain-title">🔍 Explain This Moment (t = {curr_lab_t} s)</div>
            <div class="explain-text">{moment_lab_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Render Reused Plots
        render_side_by_side_plots(lab_df_u, lab_df_p, curr_lab_t)
        render_residual_small_multiples(lab_df_p, curr_lab_t, thr)


# ===========================================================================
# TAB 2: SCENARIO EXPLORER (PHASES 1-6)
# ===========================================================================
with tab_demo:
    with st.spinner("Loading charging trajectory..."):
        sim_data = get_simulation_data(
            scenario_selection,
            ambient_c=amb_temp,
            inlet_c=inlet_temp,
            seed=seed_val,
            start_s=start_s_val,
            use_cache=use_saved_data and not run_clicked
        )

    df_u = sim_data["df_unpr"]
    df_p = sim_data["df_prot"]
    df_plain = sim_data["df_plain"]
    df_plain_a = sim_data["df_plain_a"]
    kpis = sim_data["kpis"]
    gate_events = sim_data.get("gate_events", [])
    verdicts = sim_data.get("verdicts", [])

    max_time_s = int(max(
        float(df_u["time_s"].max()) if len(df_u) > 0 else 0,
        float(df_p["time_s"].max()) if len(df_p) > 0 else 0
    ))

    if "playback_time" not in st.session_state:
        st.session_state.playback_time = min(312, max_time_s)

    badge_class = kpis.get("overall_badge", "NORMAL")
    badge_style = "badge-attack" if badge_class == "ATTACK" else ("badge-fault" if badge_class == "FAULT" else "badge-normal")
    current_gate_mode = kpis.get("final_mode", "NOMINAL")

    st.markdown(f"""
    <div class="badge-container">
        <span class="scenario-title">{scenario_selection.upper()}</span>
        <span class="{badge_style}">CLASS: {badge_class}</span>
        <span class="badge-mode">FINAL MODE: {current_gate_mode}</span>
        <span class="badge-mode">BASELINE: {kpis.get('dur_norm_base', 1594):.0f}s</span>
    </div>
    """, unsafe_allow_html=True)

    col_pb1, col_pb2, col_pb3, col_pb4 = st.columns([5, 1, 1, 1])
    with col_pb1:
        scrub_time = st.slider(
            "Simulation Playback Timeline (Seconds)",
            min_value=1,
            max_value=max_time_s,
            value=min(st.session_state.playback_time, max_time_s),
            step=1,
            key="timeline_slider"
        )
        st.session_state.playback_time = scrub_time

    with col_pb2:
        if st.button("⏮ Start", key="exp_start"):
            st.session_state.playback_time = 1
            st.rerun()
    with col_pb3:
        if st.button("⚡ Attack (300s)", key="exp_atk"):
            st.session_state.playback_time = min(300, max_time_s)
            st.rerun()
    with col_pb4:
        if st.button("⏭ End", key="exp_end"):
            st.session_state.playback_time = max_time_s
            st.rerun()

    curr_t = st.session_state.playback_time
    idx_p = (df_p["time_s"] - curr_t).abs().argmin() if len(df_p) > 0 else 0
    row_p = df_p.iloc[idx_p] if len(df_p) > 0 else pd.Series()

    verdict_curr = verdicts[idx_p] if verdicts and idx_p < len(verdicts) else None
    recent_event = None
    for ev in reversed(gate_events):
        if ev.get("t", 0) <= curr_t:
            recent_event = ev
            break

    moment_text = explain_moment(curr_t, row_p, verdict_curr, recent_event)

    st.markdown(f"""
    <div class="explain-box">
        <div class="explain-title">🔍 Explain This Moment (t = {curr_t} s)</div>
        <div class="explain-text">{moment_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Explorer KPIs
    col_kpi1, col_kpi2, col_kpi3 = st.columns([1, 1, 1])
    with col_kpi1:
        st.markdown('<div class="col-header-unpr">UNPROTECTED (NO GATE)</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Peak Temp</div>
                <div class="kpi-value {'kpi-danger' if kpis['peak_t_unpr'] > 45.0 else 'kpi-safe'}">{kpis['peak_t_unpr']:.1f} °C</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Time &gt;45°C</div>
                <div class="kpi-value {'kpi-danger' if kpis['s45_unpr'] > 0 else 'kpi-safe'}">{kpis['s45_unpr']} s</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Time &gt;50°C</div>
                <div class="kpi-value {'kpi-danger' if kpis['s50_unpr'] > 0 else 'kpi-safe'}">{kpis['s50_unpr']} s</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Charge Time</div>
                <div class="kpi-value">{kpis['dur_unpr']:.0f} s</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Final SOC</div>
                <div class="kpi-value">{kpis['soc_unpr']*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        st.markdown('<div class="col-header-baseline">PLAIN-ALARM BASELINE (MAX TEMP &ge; 45°C)</div>', unsafe_allow_html=True)
        plain_alarm_status = "TRIPPED" if kpis["plain_tripped"] else "NOT TRIPPED"
        plain_status_class = "kpi-warning" if kpis["plain_tripped"] else ("kpi-danger" if kpis["peak_t_plain"] > 45 else "kpi-safe")
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Trip Status</div>
                <div class="kpi-value {plain_status_class}">{plain_alarm_status}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Peak Temp</div>
                <div class="kpi-value {'kpi-danger' if kpis['peak_t_plain'] > 45.0 else 'kpi-safe'}">{kpis['peak_t_plain']:.1f} °C</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Time &gt;45°C</div>
                <div class="kpi-value {'kpi-danger' if kpis['s45_plain'] > 0 else 'kpi-safe'}">{kpis['s45_plain']} s</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Charge Time</div>
                <div class="kpi-value">{kpis['dur_plain']:.0f} s</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">A-Only Baseline</div>
                <div class="kpi-value {'kpi-danger' if not kpis['plain_a_tripped'] and kpis['peak_t_plain_a'] > 45 else 'kpi-safe'}">{'TRIPPED' if kpis['plain_a_tripped'] else 'BLINDED'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown('<div class="col-header-prot">PHYSICSGUARD PROTECTED (GATE + TWIN)</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Peak Temp</div>
                <div class="kpi-value {'kpi-safe' if kpis['peak_t_prot'] <= 45.0 else 'kpi-danger'}">{kpis['peak_t_prot']:.1f} °C</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Time &gt;45°C</div>
                <div class="kpi-value kpi-safe">{kpis['s45_prot']} s</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Time &gt;50°C</div>
                <div class="kpi-value kpi-safe">{kpis['s50_prot']} s</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Duration (% Delta)</div>
                <div class="kpi-value">{kpis['dur_prot']:.0f} s ({kpis['penalty_pct']:+3.1f}%)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Final SOC</div>
                <div class="kpi-value kpi-safe">{kpis['soc_prot']*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Explorer Plots
    render_side_by_side_plots(df_u, df_p, curr_t)
    render_residual_small_multiples(df_p, curr_t, thr)


# ===========================================================================
# TAB 3: HOW IT WORKS
# ===========================================================================
with tab1:
    st.markdown("### Closed-Loop Cyber-Physical Defense Architecture")
    col_arch1, col_arch2 = st.columns([1.2, 1.0])
    with col_arch1:
        st.markdown("""
```
                [ Sensor Bus (CAN / Telematics) ]  <-- Untrusted Adversary (Spoof/Drift/Freeze)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       SafetyGate.sanitize()                  │  <-- Trusted Hardware Gateway (SIL-3)
        │  • Validates Redundancy: |Ta - Tb| <= 1.44°C │
        │  • Checks Digital Twin Error: |Ta - Ttwin|   │
        │  • Virtual Substitution on Fault/Attack      │
        └──────────────────────────────────────────────┘
                               │ (Sanitized Trusted Temperatures)
                               ▼
        ┌──────────────────────────────────────────────┐
        │       ChargerBMS Controller                  │  <-- Upstream Software (Untrusted / Compromised)
        │  • Compute Base Current & Thermal Derating   │
        └──────────────────────────────────────────────┘
                               │ (Requested Current & Pump Commands)
                               ▼
        ┌──────────────────────────────────────────────┐
        │       SafetyGate.screen()                    │  <-- Trusted Actuator Boundary
        │  • Hard Clamp <= 130.56 A Envelope           │
        │  • Mandatory Pump Interlock (>= 10 L/min)    │
        │  • 180s Look-Ahead Bisection Search          │
        └──────────────────────────────────────────────┘
                               │ (Applied Safe Commands)
                               ▼
                [ Physical Plant Contactors & Pump ]
```
        """)
    with col_arch2:
        st.markdown(r"""
        #### Core Physical Guarantees
        1. **Virtual Sensor Layer (`sanitize`)**:
           - **Single Sensor Blamed**: Instantly replaces bad sensor with redundant channel adjusted for thermal gradient ($T_{sub} = T_b - 0.25^\circ\text{C}$).
           - **Both Sensors Blamed**: Replaces both readings with Digital Twin prediction $+3.0^\circ\text{C}$ conservative margin.
        2. **Actuator Boundary Layer (`screen`)**:
           - **Pump Interlock**: Whenever charge current $> 20\text{ A}$, cooling flow is forced to $\ge 10\text{ L/min}$.
           - **Thermal Lookahead**: Simulates twin $180\text{ s}$ ahead. If predicted $T_{batt} > 45.0^\circ\text{C}$, current is scaled down until safe ($\le 44.0^\circ\text{C}$).
        3. **Hardware Placement Assumption**:
           - SafetyGate runs on isolated, hardwired SIL-3 / ISO 26262 compliant gateway hardware at the physical contactor boundary, immune to network attacks.
        """)


# ===========================================================================
# TAB 4: PHYSICS RESIDUAL MATRIX
# ===========================================================================
with tab2:
    st.markdown("### Multi-Channel Physics Residual Isolation Matrix")
    st.caption("Cached mapping of which physical conservation laws trigger for each scenario class (seed 42).")

    res_matrix_data = []
    for s_name in ALL_SCENARIOS:
        f_p = DEMO_CACHE_DIR / f"{s_name}_protected.csv"
        if f_p.exists():
            df_item = pd.read_csv(f_p)
            max_r1a = df_item["R1a"].abs().max() if "R1a" in df_item.columns else 0.0
            max_r1b = df_item["R1b"].abs().max() if "R1b" in df_item.columns else 0.0
            max_r2 = df_item["R2"].abs().max() if "R2" in df_item.columns else 0.0
            max_r3 = df_item["R3"].abs().max() if "R3" in df_item.columns else 0.0
            max_rv = df_item["Rv"].abs().max() if "Rv" in df_item.columns else 0.0
            max_rflow = df_item["Rflow"].abs().max() if "Rflow" in df_item.columns else 0.0
            max_rcur = df_item["Rcur"].abs().max() if "Rcur" in df_item.columns else 0.0

            res_matrix_data.append({
                "Scenario": s_name,
                "R1a (Temp Twin)": f"{max_r1a:.1f} °C {'🔴' if max_r1a > 5.04 else '🟢'}",
                "R1b (Temp Twin)": f"{max_r1b:.1f} °C {'🔴' if max_r1b > 5.04 else '🟢'}",
                "R2 (Energy Bal)": f"{max_r2:.1f} kJ {'🔴' if max_r2 > 4.91 else '🟢'}",
                "R3 (|A - B|)": f"{max_r3:.1f} °C {'🔴' if max_r3 > 1.44 else '🟢'}",
                "Rv (Voltage)": f"{max_rv:.1f} V {'🔴' if max_rv > 10.61 else '🟢'}",
                "Rflow (Hydraulic)": f"{max_rflow:.1f} L/m {'🔴' if max_rflow > 0.55 else '🟢'}",
                "Rcur (Current)": f"{max_rcur:.1f} A {'🔴' if max_rcur > 10.0 else '🟢'}",
            })

    df_mat_view = pd.DataFrame(res_matrix_data)
    st.dataframe(df_mat_view, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 5: EVALUATION RESULTS & HELDOUT SUITE
# ===========================================================================
with tab3:
    st.markdown("### Formal Evaluation & Held-Out Statistical Verification")
    st.info("💡 **Key Finding**: Coordinated on a hot day completes safely but takes 6921 s because both sensors are spoofed (safety over speed).")

    res_sec1, res_sec2 = st.columns(2)

    with res_sec1:
        st.markdown("#### Attack Detection Performance (200 Held-Out Sessions)")
        st.caption("Wilson 95% Score Confidence Intervals across fresh seeds 700–899 with random parameter perturbations.")
        
        heldout_f = ROOT_DIR / "outputs" / "heldout_report_v2.csv"
        if heldout_f.exists():
            df_held = pd.read_csv(heldout_f)
            df_att = df_held[df_held["metric_type"] == "attack_detection"].copy()
            df_att_display = df_att[[
                "class_or_type", "protected_count", "protected_ci_95", "median_lat_s", "p95_lat_s"
            ]].rename(columns={
                "class_or_type": "Attack Vector",
                "protected_count": "Detection Count",
                "protected_ci_95": "95% Wilson CI",
                "median_lat_s": "Median Latency (s)",
                "p95_lat_s": "p95 Latency (s)"
            })
            st.dataframe(df_att_display, use_container_width=True, hide_index=True)

    with res_sec2:
        st.markdown("#### Thermal Safety Rates")
        st.caption("300 evaluation sessions under extreme attack and fault conditions.")
        if heldout_f.exists():
            df_therm = df_held[df_held["metric_type"] == "thermal_safety"].copy()
            df_therm_disp = df_therm[[
                "class_or_type", "protected_count", "protected_ci_95", "unprotected_count", "unprotected_ci_95"
            ]].rename(columns={
                "class_or_type": "Safety Metric",
                "protected_count": "Protected Safe Count",
                "protected_ci_95": "Protected 95% CI",
                "unprotected_count": "Unprotected Safe Count",
                "unprotected_ci_95": "Unprotected 95% CI"
            })
            st.dataframe(df_therm_disp, use_container_width=True, hide_index=True)

    st.markdown("#### Extreme Hot-Day Attacks (35°C Ambient / 35°C Coolant Inlet)")
    hot_summary_f = ROOT_DIR / "outputs" / "hot_protection_summary.csv"
    if hot_summary_f.exists():
        df_hot = pd.read_csv(hot_summary_f)
        df_hot_clean = df_hot[[
            "scenario", "mode", "sample_count", "peak_temp_median_c", "peak_temp_worst_c",
            "seconds_above_45C_worst", "charge_time_s_median"
        ]].rename(columns={
            "scenario": "Scenario",
            "mode": "Mode",
            "sample_count": "Runs",
            "peak_temp_median_c": "Median Peak (°C)",
            "peak_temp_worst_c": "Worst Peak (°C)",
            "seconds_above_45C_worst": "Worst Time >45°C (s)",
            "charge_time_s_median": "Median Duration (s)"
        })
        st.dataframe(df_hot_clean, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 6: LIMITATIONS & TECHNICAL ASSUMPTIONS
# ===========================================================================
with tab4:
    st.markdown("### Limitations & Engineering Assumptions")

    hot_day_norm_base = 1784.0
    coordinated_hot_time = 6921.0
    
    hot_attacks_f = ROOT_DIR / "outputs" / "hot_day_attacks.csv"
    if hot_attacks_f.exists():
        try:
            df_ha = pd.read_csv(hot_attacks_f)
            coord_row = df_ha[(df_ha["scenario"] == "coordinated") & (df_ha["mode"] == "protected")]
            if len(coord_row) > 0:
                coordinated_hot_time = float(coord_row["charge_time_s"].iloc[0])
        except Exception:
            pass

    prot_vs_unpr_f = ROOT_DIR / "outputs" / "protected_vs_unprotected.csv"
    if prot_vs_unpr_f.exists():
        try:
            df_pvu = pd.read_csv(prot_vs_unpr_f)
            hot_norm_row = df_pvu[df_pvu["scenario"] == "hot_day"]
            if len(hot_norm_row) > 0:
                hot_day_norm_base = float(hot_norm_row["charge_time_normal_baseline"].iloc[0])
        except Exception:
            pass

    coord_ratio = coordinated_hot_time / hot_day_norm_base if hot_day_norm_base > 0 else 3.88
    coord_penalty_pct = ((coordinated_hot_time - hot_day_norm_base) / hot_day_norm_base) * 100.0

    st.markdown(rf"""
    #### Engineering Trade-offs & Limitations
    1. **Safety vs Charging Speed Trade-off**:
       Under a coordinated stealth attack on a hot day ($35^\circ\text{C}$ ambient / $35^\circ\text{C}$ coolant inlet), both redundant temperature sensors are compromised and flow feedback is faked. 
       PhysicsGuard detects temperature-model discrepancies and switches to `LIMP_HOME` mode ($25\text{ A}$ current cap).
       - **Protected Session Duration**: **{coordinated_hot_time:.0f} s**
       - **Matching Normal Hot-Day Baseline**: **{hot_day_norm_base:.0f} s**
       - **Measured Ratio**: **{coord_ratio:.2f}×** (+{coord_penalty_pct:.1f}% charge time penalty).
       *In the absence of trusted thermal feedback, prioritizing physical safety prevents thermal runaway at the expense of charging speed.*
    2. **Simulation Only**: Validated in high-fidelity mathematical simulation against empirical NASA B0005 cell thermal profiles, rather than live commercial vehicle packs.
    3. **Lumped Thermal Model**: Digital twin uses a lumped single-node thermal mass model representative of pack bulk average temperature; it does not model intra-module spatial gradients.
    4. **Actuator Boundary Trust**: Assumes physical wiring between SafetyGate and power contactors/pump is intact (SIL-3 hardware placement).
    """)

    st.markdown("#### Parameter & Source Justification Table")
    source_table_data = [
        {"Parameter": "Cell Model", "Value": "Samsung INR18650-30Q", "Unit": "-", "Basis": "SOURCE (Samsung SDI Datasheet)"},
        {"Parameter": "Pack Architecture", "Value": "89s 32p (2,848 cells)", "Unit": "-", "Basis": "ASSUMPTION (320 V nominal, 30.25 kWh)"},
        {"Parameter": "Rated Charge Current", "Value": "128.0", "Unit": "A", "Basis": "SOURCE (32p x 4.0 A continuous charge rating)"},
        {"Parameter": "Max Current Envelope", "Value": "130.56", "Unit": "A", "Basis": "Calculated/derived (128.0 A x 1.02 tolerance)"},
        {"Parameter": "Twin Temp Margin", "Value": "3.0", "Unit": "°C", "Basis": "ASSUMPTION/derived (p99 error + buffer)"},
        {"Parameter": "Instant Current Cap (Rcur)", "Value": "10.0", "Unit": "A", "Basis": "ASSUMPTION/derived (Hand-set 10 A safety cap)"},
        {"Parameter": "Pump Interlock Flow", "Value": "10.0", "Unit": "L/min", "Basis": "ASSUMPTION (Automotive fast-charge cooling flow)"},
        {"Parameter": "Lookahead Horizon", "Value": "180", "Unit": "s", "Basis": "ASSUMPTION (3-minute thermal predictive window)"},
        {"Parameter": "Limp-Home Current", "Value": "25.0", "Unit": "A", "Basis": "ASSUMPTION (Reduces I²R Joule heating by >96%)"},
    ]
    st.table(pd.DataFrame(source_table_data))
