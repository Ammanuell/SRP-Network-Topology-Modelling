import json
import opendssdirect as dss
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import random
import time


def modify_circuit_voltage(vsource_pu):
    """
    Modify the circuit source voltage at each timestep

    Parameters:
        vsource_pu: numpy array of shape (n_timepoints,) containing per-unit voltage values
    """
    dss.Circuit.SetActiveElement("Vsource.Source")
    dss.Properties.Value("pu", str(vsource_pu))


def generate_transformer_voltage_profile(n_points, base_pu=1.0,
                                         daily_variation=0.02,
                                         noise_std=0.005):
    """
    Generate a realistic transformer voltage profile with daily variations

    Parameters:
        n_points: Number of timepoints to generate
        base_pu: Base per-unit voltage (default 1.0)
        daily_variation: Magnitude of daily variation (default 0.02 p.u.)
        noise_std: Standard deviation of random noise (default 0.005 p.u.)

    Returns:
        numpy array of shape (n_points,) containing voltage values in p.u.
    """
    # Time vector (assuming points represent minutes)
    t = np.linspace(0, 2 * np.pi * (n_points / 288), n_points)  # 1440 minutes per day

    # Daily pattern (higher during day, lower at night)
    daily_pattern = base_pu + daily_variation * np.sin(t - np.pi / 3)

    # Add random noise
    noise = np.random.normal(0, noise_std, n_points)

    # Combine patterns and ensure voltage stays within reasonable bounds
    voltage = daily_pattern + noise
    voltage = np.clip(voltage, base_pu - 0.1, base_pu + 0.1)

    return voltage


def generate_synthetic_power(n_customers=11, n_points=1000):
    """
    Generate synthetic P and Q data using basic daily patterns plus noise
    """
    # Create time vector (assuming samples represent minutes)
    t = np.linspace(0, 2 * np.pi, 288)  # One day

    # Base daily pattern (higher during day, lower at night)
    daily_pattern = 0.6 + 0.4 * np.sin(t - np.pi / 3)

    # Replicate for all days
    n_days = n_points // 288 + 1
    base_pattern = np.tile(daily_pattern, n_days)[:n_points]

    # Generate P and Q for each customer
    P = np.zeros((n_customers, n_points))
    Q = np.zeros((n_customers, n_points))

    for i in range(n_customers):
        # Add randomness to base pattern for each customer
        customer_scale = np.random.uniform(0.8, 1.2)  # Random scaling
        noise = np.random.normal(0, 0.1, n_points)  # Random noise
        weekly_var = np.sin(np.linspace(0, 2 * np.pi * n_days / 7, n_points)) * 0.4  # Weekly variation

        # Generate P (active power)
        P[i] = customer_scale * base_pattern + noise + weekly_var
        P[i] = np.abs(P[i])  # Ensure positive values

        # Generate Q (reactive power) with typical power factor
        pf = np.random.uniform(0.85, 0.95)  # Random power factor
        Q[i] = P[i] * np.sqrt(1 / pf ** 2 - 1) * np.random.uniform(0.9, 1.1, n_points)

    return P, Q


# Example usage:
def load_matlab_data(p_path, q_path):
    """
    Load P and Q matrices from MATLAB .mat files
    Returns:
        P, Q: numpy arrays of shape (n_customers, n_timestamps)
    """
    from scipy.io import loadmat

    P_data = loadmat(p_path)
    Q_data = loadmat(q_path)

    # Assuming the variables are named 'P' and 'Q' in the .mat files
    return P_data['P'], Q_data['Q']



class JSONtoDSS:
    def __init__(self, json_file=None):
        """Initialize converter with optional JSON file"""
        self.dss_commands = []
        if json_file:
            self.load_json(json_file)

    def load_json(self, json_file):
        """Load JSON file or string"""
        if isinstance(json_file, str):
            if Path(json_file).exists():
                with open(json_file, 'r') as f:
                    self.network_data = json.load(f)
            else:
                self.network_data = json.loads(json_file)
        else:
            self.network_data = json_file

    def create_circuit(self):
        """Create circuit command"""
        circuit = self.network_data['LVNetwork']['Circuit']
        cmd = (f"New Circuit.{circuit['name']} "
               f"basekv=11 "
               f"pu={circuit['pu']} "
               f"angle=0 "
               f"frequency=50 "
               f"phases={circuit['phases']}")
        self.dss_commands.append(cmd)

        self.dss_commands.append(f"set datapath=C:/Users/Amman/OneDrive/Documents/Uni/SRP/OpenStreetMap")

    def create_transformer(self):
        """Create transformer command"""
        xfmr = self.network_data['LVNetwork']['Transformer']
        cmd = (f"New Transformer.{xfmr['name']} "
               f"phases={xfmr['phases']} "
               f"windings={xfmr['windings']} "
               f"buses=(Sourcebus, B_1) "
               f"conns=(delta, wye) "
               f"kvs=(11, 0.415) "
               f"kvas=(250, 250) "
               f"%loadloss=0 xhl=2.5")
        self.dss_commands.append(cmd)

    def create_lines(self):
        """Create line commands"""
        for line in self.network_data['LVNetwork']['Lines']:

            if line['phases']==0:

                cmd = (f"New Line.{line['name']} "
                       f"bus1={line['bus1']}.{line['p_connect']} "
                       f"bus2={line['bus2']}.{line['p_connect']} "
                       f"length={line['length_km']} "
                       f"units=km "
                       f"linecode=35mm")
            else:
                cmd = (f"New Line.{line['name']} "
                       f"bus1={line['bus1']} "
                       f"bus2={line['bus2']} "
                       f"length={line['length_km']} "
                       f"units=km "
                       f"linecode=95mm")

            self.dss_commands.append(cmd)

    def create_loads(self):
        """Create load commands"""
        for load in self.network_data['LVNetwork']['Loads']:
            # Calculate kW from kVAR and power factor
            kvar = load['kVAR']
            pf = load['pf']
            kw = kvar * pf / (1 - pf * pf) ** 0.5

            cmd = (f"New Load.{load['customer']} "
                   f"bus1={load['bus']}.{load['phase_connect']} "
                   f"phases=1 "
                   f"kV={load['kV']} "
                   f"kW={kw:.3f} "
                   f"pf={load['pf']} "
                   f"status=fixed")
            self.dss_commands.append(cmd)

    def generate_dss(self):
        """Generate all DSS commands"""
        self.dss_commands = []  # Clear existing commands

        # Initial commands
        self.dss_commands.append("Clear")
        self.dss_commands.append("Set DefaultBaseFrequency=50")

        self.create_circuit()

        self.dss_commands.append("new linecode.95mm nphases=3 R1=0.322 X1=0.074 R0=1.282 X0=0.125 units=km")
        self.dss_commands.append("new linecode.35mm nphases=1 R1=0.868 X1=0.077 R0=0.910 X0=0.077 units=km")
        #self.dss_commands.append("new linecode.35mm nphases=1 R1=0.001 X1=0.001 R0=0.001 X0=0.001 units=km")
        #self.dss_commands.append("new linecode.95mm nphases=3 R1=1.000 X1=0.070 R0=1.000 X0=0.100 units=km")

        self.create_transformer()
        self.create_lines()
        self.create_loads()

        # Monitoring
        self.dss_commands.extend([
            "New Monitor.MonLine1 element=Line.Line1 terminal=1",
            "New Monitor.MonLoad1 element=Load.Load1 terminal=1"
        ])

        # Solution setup
        (self.dss_commands.extend([
            "Set voltagebases=[11, 0.415]",
            "Calcvoltagebases",
            "",  # Empty line for readability
            "solve mode=snap maxiterations=100"]))
        #,)
        #    "show voltages LN Nodes",
        #    "show currents residual=yes elements=yes",
        #])

        return self.dss_commands

    def save_to_file(self, filename):
        """Save DSS commands to file"""
        commands = self.generate_dss()
        with open(filename, 'w') as f:
            f.write('\n'.join(commands))

    def direct_run(self):
        """Run the network directly using OpenDSSDirect"""
        commands = self.generate_dss()

        # Start fresh
        dss.Command('Clear')

        print("\nExecuting commands:")
        # Execute each command
        for cmd in commands:
            if cmd.strip():  # Skip empty lines
                print(f"Executing: {cmd}")
                dss.Command(cmd)

                # After circuit creation, enable it
                if "New Circuit" in cmd:
                    if not dss.Circuit.Name():
                        print("Warning: Circuit not properly created!")

                # After solving, check convergence
                if "solve" in cmd.lower():
                    if not dss.Solution.Converged():
                        print("Warning: Solution did not converge!")

        return dss.Solution.Converged()


def run_time_series_powerflow(P_series, Q_series, V_transformer=None):
    """
    Modified power flow function with transformer voltage variations

    Parameters:
        P_series: numpy array of shape (n_timepoints, n_loads) containing P values in kW
        Q_series: numpy array of shape (n_timepoints, n_loads) containing Q values in kVAr
        V_transformer: numpy array of shape (n_timepoints,) containing transformer voltage in p.u.
                     If None, uses default infinite bus

    Returns:
        numpy array of shape (n_timepoints, n_buses, n_phases) containing voltages
    """
    n_timepoints, n_loads = P_series.shape
    voltages = []
    transformer_powers = []  # Store transformer P and Q values

    # If no transformer voltage profile provided, create constant one
    if V_transformer is None:
        V_transformer = np.ones(n_timepoints)

    # For each time point
    for t in range(n_timepoints):
        # Update transformer voltage
        modify_circuit_voltage(V_transformer[t])

        # Update all load values
        for i in range(n_loads):
            load_name = f"Load{i + 1}"
            dss.Loads.Name(load_name)
            dss.Loads.kW(P_series[t, i])
            dss.Loads.kvar(Q_series[t, i])

        # Solve power flow
        dss.Solution.Solve()

        if not dss.Solution.Converged():
            print(f"Warning: Solution did not converge at time {t}")

        # Get transformer powers
        dss.Circuit.SetActiveElement("Transformer.LV_Transformer")  # Use your transformer name here
        powers = dss.CktElement.Powers()
        # powers contains [P1, Q1, P2, Q2, P3, Q3] for each terminal
        # We'll take the total power from the primary side (first terminal)
        P_transformer = powers[8:13:2]#powers[0:5:2]  # Sum of P1, P2, P3
        Q_transformer = powers[9:14:2]#powers[1:6:2]  # Sum of Q1, Q2, Q3
        transformer_powers.append(P_transformer + Q_transformer)

        # Get voltages for all buses at this time step
        time_voltages = np.zeros((len(dss.Circuit.AllBusNames())*3))
        for bus in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus)
            v = dss.Bus.Voltages()
            v_mag = np.array([abs(complex(v[i], v[i + 1])) for i in range(0, len(v), 2)])

            v_mag = v_mag.tolist()
            while len(v_mag) < 3:
                v_mag.append(0)

            try:
                t_loc = int(bus[2:])
            except ValueError:
                t_loc = 0

            time_voltages[t_loc * 3: t_loc * 3 + 3] = v_mag

        voltages.append(time_voltages.tolist())

    return np.array(voltages), np.array(transformer_powers)
def analyze_network():
    """Analyze network after solving"""
    print("\nVoltage Report:")
    print("==============")
    buses = dss.Circuit.AllBusNames()


    for bus in buses:
        dss.Circuit.SetActiveBus(bus)
        v_mag_ang = dss.Bus.puVmagAngle()
        print(f"\nBus {bus}:")

        # v_mag_ang contains [mag1, ang1, mag2, ang2, mag3, ang3]
        for phase in range(len(v_mag_ang) // 2):
            mag = v_mag_ang[phase * 2]
            ang = v_mag_ang[phase * 2 + 1]
            print(f"  Phase {phase + 1}: {mag:.4f} pu ∠{ang:.1f}°")

    # Print losses
    losses = dss.Circuit.Losses()
    print(f"\nTotal Losses: {losses[0] / 1000:.2f} kW, {losses[1] / 1000:.2f} kVAR")

    # Print power flow
    total_power = dss.Circuit.TotalPower()
    print(f"Total Circuit Power: {-total_power[0]:.2f} kW, {-total_power[1]:.2f} kVAR")


def main(n_times, tr_model, load_profile_model, pv_pen):
    # Example usage
    t0 = time.time()
    converter = JSONtoDSS()
    print(time.time()-t0)
    # Load your network data
    with open('C:/Users/Amman/OneDrive/Documents/Uni/SRP/OpenStreetMap/network_config.json', 'r') as f:
        network_data = json.load(f)
    print(time.time() - t0)
    converter.load_json(network_data)
    print(time.time() - t0)
    # Option 1: Save to DSS file
    converter.save_to_file('network.dss')
    print("DSS file saved as 'network.dss'")
    print(time.time() - t0)
    # Option 2: Run directly with OpenDSSDirect
    print("\nRunning direct simulation...")
    if converter.direct_run():
        print("\nCircuit solved successfully!")
        analyze_network()
    else:
        print("Circuit solution did not converge")
    print(time.time() - t0)

    buses = dss.Circuit.AllBusNames()
    voltages = []
    print(time.time() - t0)
    for bus in buses:
        dss.Circuit.SetActiveBus(bus)
        v = dss.Bus.Voltages()  # Returns a list of voltages (magnitude and angle alternating)
        # Convert to L-N voltage magnitudes (every 2 entries form a complex number)
        v_mag = np.array([abs(complex(v[i], v[i + 1])) for i in range(0, len(v), 2)])
        voltages.append(v_mag)
    print(time.time() - t0)
    #voltages = np.array(voltages)

    # Assuming you have 11 loads and want to simulate 24 time points
    n_loads = len(network_data["LVNetwork"]["Loads"])

    # Example: Create some random P and Q profiles
    # In practice, you'd load these from your CSV or other data source
    T = n_times


    if load_profile_model==0:
        #P_profiles, Q_profiles = generate_synthetic_power(n_customers=n_loads, n_points=n_times)

        #P_profiles = np.transpose(P_profiles)
        #Q_profiles = np.transpose(Q_profiles)

        P1 = pd.read_csv('load_profile_5_min_interval_month.csv', nrows=T)
        G1 = pd.read_csv('PV_profile_5_min_interval_month.csv', usecols=[i for i in range(T)])

        P1 = P1.to_numpy()
        G1 = G1.to_numpy()
        G1 = np.transpose(G1)

        P = P1[:, np.random.randint(0, 99, size=n_loads)]
        G = G1[:, np.random.randint(0, 99, size=n_loads)]

        for solar_allow in range(n_loads):
            if random.random()>pv_pen:
                G[:, solar_allow] = 0 * G[:, solar_allow]

        P_total = P - G

        Q = 0 * P_total
        for i in range(n_loads):
            pf = np.random.uniform(0.85, 0.95)  # Random power factor
            Q[:, i] = P_total[:, i] * np.sqrt(1 / pf ** 2 - 1) * np.random.uniform(0.9, 1.1, T)

        P_profiles = 1*P_total


        P_profiles = P_total
        Q_profiles = Q



    else:
        # Load actual data
        profiles_V1 = pd.read_csv('Voltage_G1.csv', usecols=[i for i in range(T + 10)])
        profiles_V2 = pd.read_csv('Voltage_G2.csv', usecols=[i for i in range(T + 10)])
        profiles_V3 = pd.read_csv('Voltage_G3.csv', usecols=[i for i in range(T + 10)])
        profiles_V = profiles_V1.iloc[:, 1:(T+1)].to_numpy()
        profiles_V = np.vstack(
            (profiles_V, profiles_V2.iloc[:, 1:(T+1)].to_numpy(), profiles_V3.iloc[:, 1:(T+1)].to_numpy())
        )

        profiles_IR1 = pd.read_csv('Current_RE_G1.csv', usecols=[i for i in range(T + 10)])
        profiles_IR2 = pd.read_csv('Current_RE_G2.csv', usecols=[i for i in range(T + 10)])
        profiles_IR3 = pd.read_csv('Current_RE_G3.csv', usecols=[i for i in range(T + 10)])
        profiles_IR = profiles_IR1.iloc[:, 1:(T+1)].to_numpy()
        profiles_IR = np.vstack(
            (profiles_IR, profiles_IR2.iloc[:, 1:(T + 1)].to_numpy(), profiles_IR3.iloc[:, 1:(T + 1)].to_numpy())
        )

        profiles_IX1 = pd.read_csv('Current_IM_G1.csv', usecols=[i for i in range(T + 10)])
        profiles_IX2 = pd.read_csv('Current_IM_G2.csv', usecols=[i for i in range(T + 10)])
        profiles_IX3 = pd.read_csv('Current_IM_G3.csv', usecols=[i for i in range(T + 10)])
        profiles_IX = profiles_IX1.iloc[:, 1:(T+1)].to_numpy()
        profiles_IX = np.vstack(
            (profiles_IX, profiles_IX2.iloc[:, 1:(T + 1)].to_numpy(), profiles_IX3.iloc[:, 1:(T + 1)].to_numpy())
        )

        P_profiles = profiles_V * profiles_IR
        Q_profiles = profiles_V * profiles_IX

        start_height = P_profiles.shape[0]

        # If we have more rows than n_loads, remove rows randomly
        while P_profiles.shape[0] > n_loads:
            # Randomly select a row index to delete
            r_i = random.randrange(0, P_profiles.shape[0])

            # Remove the row from both P_profiles and Q_profiles
            P_profiles = np.delete(P_profiles, r_i, axis=0)
            Q_profiles = np.delete(Q_profiles, r_i, axis=0)

        while P_profiles.shape[0] < (n_loads):
            r_i = random.randrange(0, start_height)
            P_profiles = np.vstack((P_profiles, P_profiles[r_i, :]))
            Q_profiles = np.vstack((Q_profiles, Q_profiles[r_i, :]))

        permution_matrix = np.random.permutation(P_profiles.shape[0])
        P_profiles = P_profiles[permution_matrix]
        Q_profiles = Q_profiles[permution_matrix]

        P_profiles = np.transpose(P_profiles) / 1000
        Q_profiles = np.transpose(Q_profiles) / 1000



    if tr_model==0:
        V_transformer = generate_transformer_voltage_profile(
            n_points=n_times,
            base_pu=1.0,
            daily_variation=0,  # ±2% daily variation
            noise_std=0  # 0.5% random noise
        )

    elif tr_model==1:
        V_transformer = generate_transformer_voltage_profile(
            n_points=n_times,
            base_pu=1.0,
            daily_variation=0.02,  # ±2% daily variation
            noise_std=0.005  # 0.5% random noise
        )

    else:
        file_path = r"Synthetic_TR.mat"
        mat_data = loadmat(file_path)
        V_transformer = mat_data["VTR_clean"].T[0]/240



    # Run the time series simulation
    voltage_results, transformer_powers = run_time_series_powerflow(P_profiles, Q_profiles, V_transformer)

    # Add noise to all the customer values
    #voltage_results[:, 6:] = voltage_results[:, 6:] + np.random.normal(0, 0.5, voltage_results[:, 6:].shape)


    all_bus = dss.Circuit.AllBusNames()
    all_loads = [[0,0,0] for i in range(len(all_bus))]
    for b in all_bus:
        dss.Circuit.SetActiveBus(b)
        loads_this = dss.Bus.LoadList()
        if len(loads_this)>0:
            load_curr = [];
            for i in range(3):
                if len(loads_this)>i:
                    load_curr.append(int(loads_this[i].split("load")[1]))
                else:
                    load_curr.append(0)

            # Figure out what bus this is
            try:
                t_loc = int(b[2:])
            except ValueError:
                t_loc = 0

            all_loads[t_loc] = load_curr
        else:
            all_loads.append([0, 0, 0])


    savemat("est_results.mat", {"array": voltage_results, "all_bus": all_bus, "all_loads": np.array(all_loads), "P_profile": P_profiles, "Q_profiles": Q_profiles, "Tr_PQ": transformer_powers})

    print("Here2")



try:
    n_times
except NameError:
    n_times = 2000
    tr_model = 0
    load_profile_model = 0
    pv_pen = 0


if __name__ == "__main__":

    main(int(n_times), int(tr_model), int(load_profile_model), pv_pen)