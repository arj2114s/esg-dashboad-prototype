import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ESG & Techno-Economic Decision Framework", layout="wide")
st.title("🚛 Green Logistics: ESG & Techno-Economic Decision Framework")
st.markdown("### 📊 Scenarios: All Diesel vs. All EV vs. Smart Shift-Based Allocation")

# ==========================================
# 1. SIDEBAR INPUTS
# ==========================================
with st.sidebar:
    st.header("⚙️ Operational Parameters")

    with st.expander("⚡ Shift Tariffs & Charging", expanded=True):
        st.info("Charging Speed affects EV productivity!")
        charger_kw = st.number_input("Charger Power (kW)", value=240, step=60, help="120kW=Slow, 360kW=Ultra Fast")
        charger_eff = st.slider("Charger Efficiency (%)", 80, 100, 90) / 100

        # UPDATED TARIFFS BASED ON UPLOADED EXCEL SHEET (Weighted Averages)
        s1_price = st.number_input("Shift 1 (06:00-14:00) [₹/kWh]", value=8.44)
        s2_price = st.number_input("Shift 2 (14:00-22:00) [₹/kWh]", value=8.44)
        s3_price = st.number_input("Shift 3 (22:00-06:00) [₹/kWh]", value=7.18)
        
        shift_hours = 8
        t_load = st.number_input("Loading Time (mins)", value=30)
        t_unload = st.number_input("Unloading Time (mins)", value=30)
        tat_mins = t_load + t_unload

    # --- TRUCK PRESET DATA ---
    truck_presets = {
        "Tata Prima 2830.K (Diesel)": {
            "capex": 55.0, "maint": 4.0, "tyre": 7.2, "payload": 17.5,
            "speed_loaded": 22.0, "speed_empty": 35.0, "range": 600
        },
        "Olectra Megha 6x4 (Electric)": {
            "capex": 154.0, "maint": 2.0, "tyre": 9.0, "payload": 16.0,
            "speed_loaded": 25.0, "speed_empty": 35.0, "range": 150, "battery": 300
        },
        "Tata Prima E.28K (Electric)": {
            "capex": 135.0, "maint": 2.0, "tyre": 9.0, "payload": 18.0,
            "speed_loaded": 25.0, "speed_empty": 35.0, "range": 200, "battery": 453
        }
    }

    with st.expander("🚛 Truck Specs (User Preference)", expanded=True):

        # --- DIESEL SPECS ---
        st.markdown("### 🛢️ Diesel Truck")
        d_model_select = st.selectbox("Select Diesel Model", ["Tata Prima 2830.K (Diesel)"], index=0)
        d_data = truck_presets[d_model_select]

        c1, c2 = st.columns(2)
        with c1:
            d_payload = st.number_input("D-Payload (Tons)", value=d_data['payload'])
            d_capex = st.number_input("D-Purchase Cost (₹ Lakhs)", value=d_data['capex'])
            d_mileage_flat = st.number_input("D-Mileage (km/L)", value=2.25)
        with c2:
            d_maint_base = st.number_input("D-Maint Cost (₹/km)", value=d_data['maint'])
            d_tyre_cost = st.number_input("D-Tyre Cost (₹/km)", value=d_data['tyre'])
            d_maint_cost = d_maint_base + d_tyre_cost

        d_speed_loaded = st.number_input("D-Speed Loaded (km/h)", value=d_data['speed_loaded'])
        d_speed_empty = st.number_input("D-Speed Empty (km/h)", value=d_data['speed_empty'])
        d_price_liter = st.number_input("Diesel Price (₹/L)", value=93.0)
        d_empty_mass = 11.0

        st.markdown("---")

        # --- EV SPECS ---
        st.markdown("### ⚡ Electric Truck")
        e_model_select = st.selectbox("Select EV Model", ["Olectra Megha 6x4 (Electric)", "Tata Prima E.28K (Electric)"], index=0)
        e_data = truck_presets[e_model_select]

        c1, c2 = st.columns(2)
        with c1:
            e_payload = st.number_input("E-Payload (Tons)", value=e_data['payload'])
            e_capex = st.number_input("E-Purchase Cost (₹ Lakhs)", value=e_data['capex'])
            e_batt_capacity = st.number_input("Battery Size (kWh)", value=e_data['battery'])
        with c2:
            e_maint_base = st.number_input("E-Maint Cost (₹/km)", value=e_data['maint'])
            e_tyre_cost = st.number_input("E-Tyre Cost (₹/km)", value=e_data['tyre'])
            e_maint_cost = e_maint_base + e_tyre_cost

        e_speed_loaded = st.number_input("E-Speed Loaded (km/h)", value=e_data['speed_loaded'])
        e_speed_empty = st.number_input("E-Speed Empty (km/h)", value=e_data['speed_empty'])

        e_charge_time = (e_batt_capacity / charger_kw) * 60
        st.caption(f"⏱️ Est. Charge Time: {int(e_charge_time)} mins")

        e_range_base = st.number_input("Rated Range (km)", value=e_data['range'])
        e_empty_mass = 13.0

    with st.expander("💰 Financial & Battery Inputs", expanded=False):
        project_years = st.number_input("Project Duration (Years)", 5, 15, 10)
        batt_deg_rate = st.number_input("Battery Degradation (%/Year)", 0.0, 5.0, 2.5) / 100
        batt_repl_cost = st.number_input("Battery Replacement Cost (₹ Lakhs)", 0.0, 100.0, 60.0)
        batt_repl_year = st.number_input("Replace Battery In Year", 1, 10, 5)

    # PHYSICS CONSTANTS
    c_rr = 0.02
    eta_motor = 0.90
    eta_regen = 0.60
    grid_emission = 0.72

# ==========================================
# 2. MINE CONFIGURATION (INDIVIDUAL TARGETS)
# ==========================================
st.markdown("### ⛏️ Mine Configuration")
c1, c2, c3 = st.columns(3)

with c1:
    m1_target = st.number_input("M1 Daily Target (Tons)", value=3000, step=500)
    m1_dist = st.number_input("M1 Distance (km)", value=12.0)
    m1_slope = st.number_input("M1 Slope (%)", value=8.0)
with c2:
    m2_target = st.number_input("M2 Daily Target (Tons)", value=3000, step=500)
    m2_dist = st.number_input("M2 Distance (km)", value=25.0)
    m2_slope = st.number_input("M2 Slope (%)", value=2.0)
with c3:
    m3_target = st.number_input("M3 Daily Target (Tons)", value=3000, step=500)
    m3_dist = st.number_input("M3 Distance (km)", value=45.0)
    m3_slope = st.number_input("M3 Slope (%)", value=1.0)

# Calculate Per-Shift Targets
shift_target_m1 = m1_target / 3
shift_target_m2 = m2_target / 3
shift_target_m3 = m3_target / 3

total_target = m1_target + m2_target + m3_target

# ==========================================
# 3. PHYSICS ENGINE
# ==========================================
def get_physics_energy(mass_tons, slope_percent, distance_km, speed_kmh, is_ev):
    mass_kg = mass_tons * 1000
    slope_rad = np.arctan(slope_percent / 100)
    g = 9.81
    speed_ms = speed_kmh / 3.6
    distance_m = distance_km * 1000
    duration_sec = distance_m / speed_ms

    f_roll = mass_kg * g * c_rr * np.cos(slope_rad)
    f_grade = mass_kg * g * np.sin(slope_rad)
    f_total = f_roll + f_grade

    if is_ev:
        if f_total > 0:
            power_kw = (f_total * speed_ms) / 1000 / eta_motor
            return power_kw * (duration_sec / 3600)
        else:
            power_kw = (f_total * speed_ms) / 1000 * eta_regen
            return power_kw * (duration_sec / 3600)
    else:
        if f_total < 0: f_total = 0
        f_flat = mass_kg * g * c_rr
        effort_ratio = max(0.3, f_total / f_flat)
        base_liters = distance_km / d_mileage_flat
        return base_liters * effort_ratio

def calculate_shift_logistics(dist, slope, d_trucks, e_trucks, shift_tariff, year_idx=0):
    current_range = e_range_base * ((1 - batt_deg_rate) ** year_idx)

    t_travel_d = (dist/d_speed_loaded*60) + (dist/d_speed_empty*60)
    t_travel_e = (dist/e_speed_loaded*60) + (dist/e_speed_empty*60)
    cycle_d = t_travel_d + tat_mins
    cycle_e = t_travel_e + tat_mins

    if e_trucks > 0:
        kwh_trip = get_physics_energy(e_empty_mass+e_payload, slope, dist, e_speed_loaded, True) + \
                   get_physics_energy(e_empty_mass, -slope, dist, e_speed_empty, True)

        battery_cap = e_batt_capacity * ((1 - batt_deg_rate) ** year_idx)

        if kwh_trip > battery_cap: return None

        trip_dist = dist * 2
        if trip_dist > current_range: trips_per_charge = 1
        else: trips_per_charge = min(np.floor(battery_cap/kwh_trip) if kwh_trip>0 else 50, 100)

        charge_penalty = e_charge_time / trips_per_charge
        cycle_e_total = cycle_e + charge_penalty

        shift_mins = shift_hours * 60 - 30
        trips_per_truck_e = int(shift_mins / cycle_e_total)
        total_trips_e = e_trucks * trips_per_truck_e

        energy_cost_actual = (total_trips_e * kwh_trip * shift_tariff)
        cost_e = energy_cost_actual + \
                 (total_trips_e * dist * 2 * e_maint_cost) + \
                 (e_trucks * e_capex * 100000 / (5*300*3))

        tons_e = total_trips_e * e_payload
        co2_e = total_trips_e * kwh_trip * grid_emission
        time_e = (total_trips_e * cycle_e_total)/60
        total_kwh_e = total_trips_e * kwh_trip
    else:
        total_trips_e=0; tons_e=0; cost_e=0; co2_e=0; time_e=0; total_kwh_e=0; energy_cost_actual=0

    if d_trucks > 0:
        liters_trip = get_physics_energy(d_empty_mass+d_payload, slope, dist, d_speed_loaded, False) + \
                      get_physics_energy(d_empty_mass, -slope, dist, d_speed_empty, False)

        cycle_d_total = cycle_d
        shift_mins = shift_hours * 60 - 30
        trips_per_truck_d = int(shift_mins / cycle_d_total)
        total_trips_d = d_trucks * trips_per_truck_d

        fuel_cost_actual = (total_trips_d * liters_trip * d_price_liter)
        cost_d = fuel_cost_actual + \
                 (total_trips_d * dist * 2 * d_maint_cost) + \
                 (d_trucks * d_capex * 100000 / (5*300*3))

        tons_d = total_trips_d * d_payload
        co2_d = total_trips_d * liters_trip * 2.68
        time_d = (total_trips_d * cycle_d_total)/60
        total_liters_d = total_trips_d * liters_trip
    else:
        total_trips_d=0; tons_d=0; cost_d=0; co2_d=0; time_d=0; total_liters_d=0; fuel_cost_actual=0

    return {
        "cost": cost_d + cost_e, "co2": co2_d + co2_e, "tons": tons_d + tons_e,
        "d_trips": total_trips_d, "e_trips": total_trips_e, "time": time_d + time_e,
        "liters": total_liters_d, "kwh": total_kwh_e,
        "fuel_cost": fuel_cost_actual, "energy_cost": energy_cost_actual
    }

def get_required_trucks(dist, slope, target, tariff, v_type):
    """Helper function to find exact trucks needed for a single shift/mine"""
    if target <= 0: return 0
    for n in range(1, 150):
        if v_type == "Diesel":
            r = calculate_shift_logistics(dist, slope, n, 0, tariff)
        else:
            r = calculate_shift_logistics(dist, slope, 0, n, tariff)
        if r and r['tons'] >= target:
            return n
    return 150 # Failsafe

# ==========================================
# 4. FIXED SCENARIO CALCULATOR (UPDATED)
# ==========================================
def calculate_fixed_scenario(scenario_type):
    total_d_trucks = 0; total_e_trucks = 0
    total_cost = 0; total_co2 = 0; total_time = 0; total_tons = 0
    total_d_trips = 0; total_e_trips = 0; total_liters = 0; total_kwh = 0
    total_fuel_cost = 0; total_energy_cost = 0

    mine_configs = [
        (m1_dist, m1_slope, shift_target_m1),
        (m2_dist, m2_slope, shift_target_m2),
        (m3_dist, m3_slope, shift_target_m3)
    ]

    for tariff in [s1_price, s2_price, s3_price]:
        shift_d = 0; shift_e = 0
        for dist, slope, starget in mine_configs:
            n = get_required_trucks(dist, slope, starget, tariff, scenario_type)
            if scenario_type == "Diesel":
                r = calculate_shift_logistics(dist, slope, n, 0, tariff)
                shift_d += n
                if r:
                    total_cost+=r['cost']; total_co2+=r['co2']; total_time+=r['time']; total_tons+=r['tons']; total_d_trips+=r['d_trips']
                    total_liters += r['liters']; total_fuel_cost += r['fuel_cost']
            else:
                r = calculate_shift_logistics(dist, slope, 0, n, tariff)
                shift_e += n
                if r:
                    total_cost+=r['cost']; total_co2+=r['co2']; total_time+=r['time']; total_tons+=r['tons']; total_e_trips+=r['e_trips']
                    total_kwh += r['kwh']; total_energy_cost += r['energy_cost']

        total_d_trucks = max(total_d_trucks, shift_d)
        total_e_trucks = max(total_e_trucks, shift_e)

    return {"d_trucks": total_d_trucks, "e_trucks": total_e_trucks, "cost": total_cost, "co2": total_co2, "time": total_time,
            "tons": total_tons, "d_trips": total_d_trips, "e_trips": total_e_trips, "liters": total_liters, "kwh": total_kwh, "fuel_cost": total_fuel_cost, "energy_cost": total_energy_cost}

# ==========================================
# 5. OPTIMIZATION PROBLEM
# ==========================================
class SingleShiftProblem(ElementwiseProblem):
    def __init__(self, tariff):
        self.tariff = tariff
        super().__init__(n_var=6, n_obj=3, n_ieq_constr=3, xl=0, xu=60, vtype=int)
        
    def _evaluate(self, x, out, *args, **kwargs):
        d1, e1, d2, e2, d3, e3 = x
        r1 = calculate_shift_logistics(m1_dist, m1_slope, d1, e1, self.tariff, year_idx=0)
        r2 = calculate_shift_logistics(m2_dist, m2_slope, d2, e2, self.tariff, year_idx=0)
        r3 = calculate_shift_logistics(m3_dist, m3_slope, d3, e3, self.tariff, year_idx=0)
        
        if r1 and r2 and r3:
            out["F"] = [r1['cost']+r2['cost']+r3['cost'], r1['co2']+r2['co2']+r3['co2'], r1['time']+r2['time']+r3['time']]
            out["G"] = [
                shift_target_m1 - r1['tons'], 
                shift_target_m2 - r2['tons'], 
                shift_target_m3 - r3['tons']
            ]
        else:
            out["F"] = [1e9, 1e9, 1e9]; out["G"] = [1000, 1000, 1000]

# ==========================================
# 6. RUN SIMULATIONS & TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Main Optimization", "💰 Financial ROI", "📉 Sensitivity Analysis", "🧬 Pareto Analysis", "🔌 Infrastructure"])

with tab1:
    with st.spinner("🚀 Running Optimization..."):
        sc1 = calculate_fixed_scenario("Diesel")
        sc2 = calculate_fixed_scenario("EV")

        s1_opt = minimize(SingleShiftProblem(s1_price), NSGA2(pop_size=60), get_termination("n_gen", 40), seed=1)
        s2_opt = minimize(SingleShiftProblem(s2_price), NSGA2(pop_size=60), get_termination("n_gen", 40), seed=1)
        s3_opt = minimize(SingleShiftProblem(s3_price), NSGA2(pop_size=60), get_termination("n_gen", 40), seed=1)

        def extract_best(res, tariff):
            if len(res.X) == 0: 
                return {"cost":0, "co2":0, "time":0, "d_trucks":0, "e_trucks":0, "d_trips":0, "e_trips":0, "tons":0, "liters":0, "kwh":0, "fuel_cost":0, "energy_cost":0, 
                        "m1_d":0, "m1_e":0, "m2_d":0, "m2_e":0, "m3_d":0, "m3_e":0}
            idx = np.argsort(res.F[:, 0])[0]
            x = res.X[idx]
            r1 = calculate_shift_logistics(m1_dist, m1_slope, x[0], x[1], tariff)
            r2 = calculate_shift_logistics(m2_dist, m2_slope, x[2], x[3], tariff)
            r3 = calculate_shift_logistics(m3_dist, m3_slope, x[4], x[5], tariff)
            return {
                "cost": int(res.F[idx][0]), "co2": int(res.F[idx][1]), "time": int(res.F[idx][2]),
                "d_trucks": int(x[0]+x[2]+x[4]), "e_trucks": int(x[1]+x[3]+x[5]),
                "d_trips": int(r1['d_trips']+r2['d_trips']+r3['d_trips']),
                "e_trips": int(r1['e_trips']+r2['e_trips']+r3['e_trips']),
                "tons": int(r1['tons']+r2['tons']+r3['tons']),
                "liters": r1['liters']+r2['liters']+r3['liters'],
                "kwh": r1['kwh']+r2['kwh']+r3['kwh'],
                "fuel_cost": r1['fuel_cost']+r2['fuel_cost']+r3['fuel_cost'],
                "energy_cost": r1['energy_cost']+r2['energy_cost']+r3['energy_cost'],
                "m1_d": int(x[0]), "m1_e": int(x[1]),
                "m2_d": int(x[2]), "m2_e": int(x[3]),
                "m3_d": int(x[4]), "m3_e": int(x[5])
            }

        o1 = extract_best(s1_opt, s1_price)
        o2 = extract_best(s2_opt, s2_price)
        o3 = extract_best(s3_opt, s3_price)

        sc3 = {
            "d_trucks": max(o1['d_trucks'], o2['d_trucks'], o3['d_trucks']),
            "e_trucks": max(o1['e_trucks'], o2['e_trucks'], o3['e_trucks']),
            "cost": o1['cost']+o2['cost']+o3['cost'],
            "co2": o1['co2']+o2['co2']+o3['co2'],
            "time": o1['time']+o2['time']+o3['time'],
            "d_trips": o1['d_trips']+o2['d_trips']+o3['d_trips'],
            "e_trips": o1['e_trips']+o2['e_trips']+o3['e_trips'],
            "tons": o1['tons']+o2['tons']+o3['tons'],
            "liters": o1['liters']+o2['liters']+o3['liters'],
            "kwh": o1['kwh']+o2['kwh']+o3['kwh'],
            "fuel_cost": o1['fuel_cost']+o2['fuel_cost']+o3['fuel_cost'],
            "energy_cost": o1['energy_cost']+o2['energy_cost']+o3['energy_cost']
        }

    st.header("📊 Final Simulation Results")
    res_df = pd.DataFrame({
        "Metric": ["Diesel Fleet Size", "EV Fleet Size", "Total Trips", "Total Tons Moved", "Total CO2 (kg)", "Total Cost (₹)"],
        "Scenario 1: All Diesel": [int(sc1['d_trucks']), 0, int(sc1['d_trips']), int(sc1['tons']), int(sc1['co2']), int(sc1['cost'])],
        "Scenario 2: All EV": [0, int(sc2['e_trucks']), int(sc2['e_trips']), int(sc2['tons']), int(sc2['co2']), int(sc2['cost'])],
        "Scenario 3: Optimized": [int(sc3['d_trucks']), int(sc3['e_trucks']), int(sc3['d_trips'] + sc3['e_trips']), int(sc3['tons']), int(sc3['co2']), int(sc3['cost'])]
    })
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cost Comparison")
        fig = go.Figure(data=[go.Bar(x=["All Diesel", "All EV", "Optimized"], y=[sc1['cost'], sc2['cost'], sc3['cost']], marker_color=['#FF4B4B', '#FFB84B', '#00CC96'])])
        fig.update_layout(yaxis_title="Total Cost (₹)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Emission Comparison")
        fig2 = go.Figure(data=[go.Bar(x=["All Diesel", "All EV", "Optimized"], y=[sc1['co2'], sc2['co2'], sc3['co2']], marker_color=['#555555', '#888888', '#00CC96'])])
        fig2.update_layout(yaxis_title="CO2 (kg)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.header("🌙 Detailed Mine-Wise Deployment (All Scenarios)")
    
    # Generate detailed breakdowns dynamically for All Diesel and All EV
    shift_tariffs = [s1_price, s2_price, s3_price]
    detailed_data = []

    for s_idx, tariff in enumerate(shift_tariffs):
        shift_name = f"Shift {s_idx+1}"
        
        # Scenario 1: All Diesel
        d_m1 = get_required_trucks(m1_dist, m1_slope, shift_target_m1, tariff, "Diesel")
        d_m2 = get_required_trucks(m2_dist, m2_slope, shift_target_m2, tariff, "Diesel")
        d_m3 = get_required_trucks(m3_dist, m3_slope, shift_target_m3, tariff, "Diesel")
        detailed_data.append(["All Diesel", shift_name, d_m1, 0, d_m2, 0, d_m3, 0, d_m1+d_m2+d_m3])

        # Scenario 2: All EV
        e_m1 = get_required_trucks(m1_dist, m1_slope, shift_target_m1, tariff, "EV")
        e_m2 = get_required_trucks(m2_dist, m2_slope, shift_target_m2, tariff, "EV")
        e_m3 = get_required_trucks(m3_dist, m3_slope, shift_target_m3, tariff, "EV")
        detailed_data.append(["All EV", shift_name, 0, e_m1, 0, e_m2, 0, e_m3, e_m1+e_m2+e_m3])

        # Scenario 3: Optimized
        opt_data = [o1, o2, o3][s_idx]
        tot_opt = opt_data['m1_d']+opt_data['m1_e']+opt_data['m2_d']+opt_data['m2_e']+opt_data['m3_d']+opt_data['m3_e']
        detailed_data.append(["Optimized", shift_name, opt_data['m1_d'], opt_data['m1_e'], opt_data['m2_d'], opt_data['m2_e'], opt_data['m3_d'], opt_data['m3_e'], tot_opt])

    # Convert to DataFrame
    detailed_df = pd.DataFrame(detailed_data, columns=[
        "Scenario", "Shift", 
        "Mine 1 (Diesel)", "Mine 1 (EV)", 
        "Mine 2 (Diesel)", "Mine 2 (EV)", 
        "Mine 3 (Diesel)", "Mine 3 (EV)", 
        "Total Active Trucks"
    ])

    # Display the final large table
    st.dataframe(detailed_df.style.highlight_max(subset=["Total Active Trucks"], color='lightgreen').set_properties(**{'text-align': 'center'}), use_container_width=True, hide_index=True)

with tab2:
    st.header("💸 Financial Return on Investment")
    capex_d = sc1['d_trucks'] * d_capex * 100000
    capex_e = sc2['e_trucks'] * e_capex * 100000
    capex_opt = (sc3['d_trucks']*d_capex + sc3['e_trucks']*e_capex) * 100000

    cf_d = []; cf_e = []; cf_opt = []
    cuml_d = -capex_d; cuml_e = -capex_e; cuml_opt = -capex_opt
    days_yr = 300

    for yr in range(1, project_years + 1):
        cuml_d -= (sc1['cost'] * days_yr)
        cf_d.append(cuml_d)
        opex_e = sc2['cost'] * days_yr
        if yr == batt_repl_year: opex_e += (sc2['e_trucks'] * batt_repl_cost * 100000)
        cuml_e -= opex_e
        cf_e.append(cuml_e)
        opex_opt = sc3['cost'] * days_yr
        if yr == batt_repl_year: opex_opt += (sc3['e_trucks'] * batt_repl_cost * 100000)
        cuml_opt -= opex_opt
        cf_opt.append(cuml_opt)

    yrs = list(range(1, project_years + 1))
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=yrs, y=[int(x/1e7) for x in cf_d], mode='lines+markers', name='All Diesel', line=dict(color='red')))
    fig_roi.add_trace(go.Scatter(x=yrs, y=[int(x/1e7) for x in cf_e], mode='lines+markers', name='All EV', line=dict(color='green')))
    fig_roi.add_trace(go.Scatter(x=yrs, y=[int(x/1e7) for x in cf_opt], mode='lines+markers', name='Optimized', line=dict(color='blue')))
    fig_roi.update_layout(title="Cumulative Cash Flow (NPV Preview)", xaxis_title="Year", yaxis_title="Net Cash Position (₹ Crores)")
    st.plotly_chart(fig_roi, use_container_width=True)

with tab3:
    st.header("📉 Sensitivity Analysis")
    d_range = np.linspace(90, 140, 20)
    fixed_d = sc1['cost'] - sc1['fuel_cost']
    fixed_opt_d = sc3['cost'] - sc3['fuel_cost']
    res_d_sens = []
    res_opt_sens = []

    for p in d_range:
        res_d_sens.append(int(fixed_d + (sc1['liters'] * p)))
        res_opt_sens.append(int(fixed_opt_d + (sc3['liters'] * p)))

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=d_range, y=res_d_sens, mode='lines', name='All Diesel', line=dict(color='red')))
    fig_d.add_trace(go.Scatter(x=d_range, y=res_opt_sens, mode='lines', name='Optimized Hybrid', line=dict(color='blue')))
    fig_d.update_layout(xaxis_title="Diesel Price (₹/Liter)", yaxis_title="Daily Operating Cost (₹)")
    st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("Combined Sensitivity Heatmap")
    e_range = np.linspace(4, 18, 20)
    base_fixed_opt = sc3['cost'] - sc3['fuel_cost'] - sc3['energy_cost']
    z_data = []
    for e_p in e_range:
        row = []
        for d_p in d_range:
            row.append(int(base_fixed_opt + (sc3['liters'] * d_p) + (sc3['kwh'] * e_p)))
        z_data.append(row)
    fig_heat = go.Figure(data=go.Heatmap(z=z_data, x=d_range, y=e_range, colorscale='RdBu_r'))
    fig_heat.update_layout(xaxis_title="Diesel Price", yaxis_title="Electricity Price")
    st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    st.header("🧬 Pareto Analysis (Trade-off Visualization)")
    F = s1_opt.F
    if len(F) > 0:
        pareto_df = pd.DataFrame(F, columns=["Cost", "CO2", "Time"]).astype(int)

        st.subheader("1. 3D Solution Space")
        fig_3d = px.scatter_3d(
            pareto_df, x="Cost", y="CO2", z="Time",
            color="Cost", size_max=18, opacity=0.8,
            title="Pareto Front (Cost vs CO2 vs Time)"
        )
        fig_3d.update_layout(height=700, scene=dict(aspectmode='cube'))
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")

        st.subheader("2. Cost vs. Emissions Trade-off")
        fig_2d = px.scatter(
            pareto_df, x="Cost", y="CO2", color="Time",
            color_continuous_scale="Viridis", size_max=15,
            labels={"Cost": "Cost (₹)", "CO2": "CO2 (kg)", "Time": "Time (hrs)"}
        )
        st.plotly_chart(fig_2d, use_container_width=True)

        st.subheader("3. Strategy Comparison Flow")
        fig_par = px.parallel_coordinates(
            pareto_df, color="Cost",
            labels={"Cost": "Cost (₹)", "CO2": "CO2 (kg)", "Time": "Time (h)"},
            color_continuous_scale=px.colors.diverging.Tealrose
        )
        fig_par.update_layout(margin=dict(l=60, r=60, t=50, b=50))
        st.plotly_chart(fig_par, use_container_width=True)

with tab5:
    st.header("🔌 Infrastructure Planning")
    st.metric("Charger Power Selected", f"{charger_kw} kW")
    max_evs_active = sc3['e_trucks']
    daily_kwh = sc3['kwh']

    ports_energy = daily_kwh / (charger_kw * 24 * 0.9)
    time_charge_hrs = e_batt_capacity / charger_kw
    slots_per_shift = 8.0 / time_charge_hrs
    ports_peak = max_evs_active / slots_per_shift
    final_ports = int(np.ceil(max(ports_energy, ports_peak) * 1.1))

    c1, c2, c3 = st.columns(3)
    c1.metric("Max Active EVs", f"{max_evs_active}")
    c2.metric("Charging Time (0-100%)", f"{int(time_charge_hrs*60)} mins")
    c3.metric("Recommended Ports", f"{final_ports}", delta="Based on Peak Shift")
