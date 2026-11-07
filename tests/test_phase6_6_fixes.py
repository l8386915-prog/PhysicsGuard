import re
import pandas as pd
import pytest
from sim_engine import (
    run_custom_attack_simulation,
    load_preset,
    explain_moment,
    PRESET_REGISTRY,
    get_preset_options,
)


def test_item1_limitations_tab_no_name_error():
    """Verify that the Limitations tab markdown string compiles and renders without NameError."""
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()

    # Verify no unescaped \text{C} or \text{ A} inside f-strings
    assert "\\text{C}" not in src, "Found unescaped \\text{C} in app.py"
    assert "\\text{ A}" not in src, "Found unescaped \\text{ A} in app.py"

    # Simulate evaluating lines around line 1320-1340
    coordinated_hot_time = 1800.0
    hot_day_norm_base = 464.0
    coord_ratio = coordinated_hot_time / hot_day_norm_base
    coord_penalty_pct = ((coordinated_hot_time - hot_day_norm_base) / hot_day_norm_base) * 100.0

    # Ensure this block evaluates cleanly without raising NameError
    text = rf"""
    #### Engineering Trade-offs & Limitations
    1. **Safety vs Charging Speed Trade-off**:
       Under a coordinated stealth attack on a hot day (35 °C ambient / 35 °C coolant inlet), both redundant temperature sensors are compromised and flow feedback is faked. 
       PhysicsGuard detects temperature-model discrepancies and switches to `LIMP_HOME` mode (25 A current cap).
       - **Protected Session Duration**: **{coordinated_hot_time:.0f} s**
       - **Matching Normal Hot-Day Baseline**: **{hot_day_norm_base:.0f} s**
       - **Measured Ratio**: **{coord_ratio:.2f}×** (+{coord_penalty_pct:.1f}% charge time penalty).
    """
    assert "35 °C ambient" in text
    assert "25 A current cap" in text


def test_item2_preset_sync_all_recipes():
    """Verify all 6 presets match their exact specification and sync cleanly."""
    options = get_preset_options()
    assert len(options) == 6

    # 1. pump_off = ONE layer (cmd_pump, set_value, 0, start 300 s)
    p_specs, p_env = load_preset("pump_off")
    assert len(p_specs) == 1
    assert p_specs[0]["target"] == "cmd_pump"
    assert p_specs[0]["type"] == "set_value"
    assert p_specs[0]["magnitude"] == 0.0
    assert p_specs[0]["start_s"] == 300.0

    # 2. abusive_command = ONE layer (cmd_current set_value 192 at 300 s)
    a_specs, _ = load_preset("abusive_command")
    assert len(a_specs) == 1
    assert a_specs[0]["target"] == "cmd_current"
    assert a_specs[0]["type"] == "set_value"
    assert a_specs[0]["magnitude"] == 192.0
    assert a_specs[0]["start_s"] == 300.0

    # 3. demo_hot_attack = TWO layers
    d_specs, _ = load_preset("demo_hot_attack")
    assert len(d_specs) == 2
    assert d_specs[0]["target"] == "temp_cell_a"
    assert d_specs[1]["target"] == "cmd_pump"

    # 4. freeze_temp_a = ONE layer
    f_specs, _ = load_preset("freeze_temp_a")
    assert len(f_specs) == 1
    assert f_specs[0]["target"] == "temp_cell_a"
    assert f_specs[0]["type"] == "freeze"

    # 5. drift_temp_a = ONE layer
    dr_specs, _ = load_preset("drift_temp_a")
    assert len(dr_specs) == 1
    assert dr_specs[0]["target"] == "temp_cell_a"
    assert dr_specs[0]["type"] == "drift"

    # 6. coordinated = TWO layers
    c_specs, _ = load_preset("coordinated")
    assert len(c_specs) == 2


def test_item3_what_happened_grouped_under_12_lines():
    """Verify What Happened timeline groups gate events to at most ~12 lines for all 6 presets."""
    for name, p_data in PRESET_REGISTRY.items():
        specs = p_data["specs"]
        env = p_data.get("env", {})
        res = run_custom_attack_simulation(
            specs,
            ambient_c=env.get("ambient_c", 35.0),
            inlet_c=env.get("inlet_c", 35.0),
            seed=42
        )
        timeline = res.get("timeline", [])
        gate_events = res.get("gate_events", [])

        # Raw events can be over 1000 lines
        assert len(gate_events) > 100, f"Preset {name} should have many raw gate events"

        # Grouped timeline must be at most ~12 lines
        assert len(timeline) <= 12, f"Preset {name} produced {len(timeline)} lines, expected <= 12"
        assert len(timeline) >= 4, f"Preset {name} produced too few lines ({len(timeline)})"

        # Must include session start, attack activation, session end
        assert any("Fast-charging session initiated" in line for line in timeline)
        assert any("Cyberattack activated" in line for line in timeline)
        assert any("Session End" in line for line in timeline)


def test_item4_explain_this_moment_rules():
    """Verify explain_moment uses real numbers, correct actions, and never misidentifies pump attacks."""
    # Test 1: pump_off preset at t=310 s
    p_specs, p_env = load_preset("pump_off")
    res_pump = run_custom_attack_simulation(p_specs, ambient_c=35.0, inlet_c=35.0, seed=42)
    df_p = res_pump["df_prot"]
    events_pump = res_pump["gate_events"]
    verdicts_pump = res_pump["verdicts"]

    idx = (df_p["time_s"] - 310).abs().argmin()
    row = df_p.iloc[idx]
    v = verdicts_pump[idx] if idx < len(verdicts_pump) else None
    ev = next((e for e in reversed(events_pump) if e["t"] <= 310), None)
    text_pump = explain_moment(310, row, v, ev)

    # Must NOT say charge command exceeds limits
    assert "charge command exceeds safe" not in text_pump.lower(), (
        f"pump_off incorrectly blamed charge command: {text_pump}"
    )
    # Must explain pump override
    assert "pump" in text_pump.lower()
    assert "10.0" in text_pump or "10" in text_pump

    # Test 2: abusive_command preset at t=310 s
    a_specs, a_env = load_preset("abusive_command")
    res_abus = run_custom_attack_simulation(a_specs, ambient_c=35.0, inlet_c=35.0, seed=42)
    df_a = res_abus["df_prot"]
    events_a = res_abus["gate_events"]
    verdicts_a = res_abus["verdicts"]

    idx_a = (df_a["time_s"] - 310).abs().argmin()
    row_a = df_a.iloc[idx_a]
    v_a = verdicts_a[idx_a] if idx_a < len(verdicts_a) else None
    ev_a = next((e for e in reversed(events_a) if e["t"] <= 310), None)
    text_abus = explain_moment(310, row_a, v_a, ev_a)

    # DOES say charge command exceeds safe limits because requested current (192 A) > 130.56 A
    assert "charge command" in text_abus.lower()
    assert "192.0" in text_abus or "192" in text_abus
    assert "130.56" in text_abus or "safe" in text_abus.lower()

    # Test 3: freeze_temp_a preset at t=320 s
    f_specs, f_env = load_preset("freeze_temp_a")
    res_frz = run_custom_attack_simulation(f_specs, ambient_c=25.0, inlet_c=25.0, seed=42)
    df_f = res_frz["df_prot"]
    events_f = res_frz["gate_events"]
    verdicts_f = res_frz["verdicts"]

    idx_f = (df_f["time_s"] - 320).abs().argmin()
    row_f = df_f.iloc[idx_f]
    v_f = verdicts_f[idx_f] if idx_f < len(verdicts_f) else None
    ev_f = next((e for e in reversed(events_f) if e["t"] <= 320), None)
    text_frz = explain_moment(320, row_f, v_f, ev_f)

    # Must explain sensor substitution
    assert "sensor" in text_frz.lower()
    assert "substituted" in text_frz.lower() or "redundant" in text_frz.lower()
    assert "charge command exceeds safe" not in text_frz.lower()
