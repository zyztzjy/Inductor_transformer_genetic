import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings

warnings.filterwarnings('ignore')


class SParameterAnalyzer:
    def __init__(self, z0=50):
        self.z0 = z0

    def load_sparams(self, file_path):
        """Load S-parameter file"""
        return pd.read_csv(file_path)

    def s_to_z(self, s_params):
        """Convert S-parameters to Z-parameters"""
        n_ports = 3
        n_freq = len(s_params)
        z_params = np.zeros((n_freq, n_ports, n_ports), dtype=complex)

        for i in range(n_freq):
            S = np.zeros((n_ports, n_ports), dtype=complex)
            for p in range(n_ports):
                for q in range(n_ports):
                    mag = s_params[f'S{p + 1}_{q + 1}_mag'].iloc[i]
                    ang = np.deg2rad(s_params[f'S{p + 1}_{q + 1}_ang'].iloc[i])
                    S[p, q] = mag * np.exp(1j * ang)

            I = np.eye(n_ports)
            try:
                Z = self.z0 * np.linalg.inv(I - S) @ (I + S)
                z_params[i] = Z
            except np.linalg.LinAlgError:
                Z = self.z0 * np.linalg.pinv(I - S) @ (I + S)
                z_params[i] = Z

        return z_params

    def calculate_quality_factor(self, z_params, freq, port=0):
        """Calculate quality factor"""
        Z_in = z_params[:, port, port]
        real_part = np.real(Z_in)
        imag_part = np.imag(Z_in)

        real_part = np.where(np.abs(real_part) < 1e-12, 1e-12, real_part)
        real_part = np.where(real_part < 0, 1e-12, real_part)

        Q = np.abs(imag_part) / real_part
        Q = np.clip(Q, 0, 1000)

        return Q

    def calculate_coupling_coefficient(self, z_params, freq):
        """Calculate coupling coefficient"""
        valid_indices = freq > 1e9

        if np.sum(valid_indices) == 0:
            return np.nan

        L12 = np.imag(z_params[valid_indices, 0, 1]) / (2 * np.pi * freq[valid_indices])
        L11 = np.abs(np.imag(z_params[valid_indices, 0, 0]) / (2 * np.pi * freq[valid_indices]))
        L22 = np.abs(np.imag(z_params[valid_indices, 1, 1]) / (2 * np.pi * freq[valid_indices]))

        denominator = np.sqrt(L11 * L22)
        denominator = np.where(denominator < 1e-15, 1e-15, denominator)

        k = L12 / denominator
        k = np.clip(k, -1.0, 1.0)

        return np.mean(k)

    def calculate_bandwidth(self, s_params, freq, port_in, port_out, db_level=-3):
        """Calculate bandwidth"""
        mag_col = f'S{port_out}_{port_in}_mag'
        s_mag_db = 20 * np.log10(np.maximum(s_params[mag_col].values, 1e-12))

        max_gain = np.max(s_mag_db)
        threshold = max_gain + db_level

        above_threshold = s_mag_db >= threshold

        if np.sum(above_threshold) < 2:
            return 0

        valid_indices = np.where(above_threshold)[0]
        f_low = freq[valid_indices[0]]
        f_high = freq[valid_indices[-1]]
        bw = f_high - f_low

        return bw

    def calculate_group_delay(self, s_params, freq, port_in, port_out):
        """Calculate group delay"""
        mag_col = f'S{port_out}_{port_in}_mag'
        ang_col = f'S{port_out}_{port_in}_ang'

        sort_idx = np.argsort(freq)
        freq_sorted = freq[sort_idx]
        phase_sorted = np.unwrap(np.deg2rad(s_params[ang_col].iloc[sort_idx]))

        group_delay = -np.gradient(phase_sorted) / (2 * np.pi * np.gradient(freq_sorted))
        group_delay = np.clip(group_delay, -1e-9, 1e-9)

        return np.mean(group_delay)

    def calculate_insertion_loss(self, s_params, port_in, port_out):
        """Calculate insertion loss"""
        mag_col = f'S{port_out}_{port_in}_mag'
        s_mag_db = 20 * np.log10(np.maximum(s_params[mag_col].values, 1e-12))
        return np.min(s_mag_db)

    def analyze_file(self, file_path):
        """Analyze single file - simplified version"""
        try:
            df = self.load_sparams(file_path)
            freq = df['Frequency_Hz'].values

            z_params = self.s_to_z(df)

            results = {'file_name': os.path.basename(file_path)}

            # Use valid frequency range (1GHz-90GHz)
            valid_indices = (freq > 1e9) & (freq < 90e9)

            # 1. Quality factor (Port 1 and Port 2)
            if np.sum(valid_indices) > 0:
                Q1 = self.calculate_quality_factor(z_params, freq, 0)
                Q2 = self.calculate_quality_factor(z_params, freq, 1)
                Q1_valid = Q1[valid_indices]
                Q2_valid = Q2[valid_indices]

                results['Q_port1'] = np.mean(Q1_valid)
                results['Q_port2'] = np.mean(Q2_valid)
            else:
                results['Q_port1'] = np.nan
                results['Q_port2'] = np.nan

            # 2. Coupling coefficient (Port 1-2)
            results['coupling_coefficient'] = self.calculate_coupling_coefficient(z_params, freq)

            # 3. Bandwidth (Port 1-3 and Port 2-3)
            results['bandwidth_13_Hz'] = self.calculate_bandwidth(df, freq, 1, 3)
            results['bandwidth_23_Hz'] = self.calculate_bandwidth(df, freq, 2, 3)

            # 4. Group delay (Port 1-3 and Port 2-3)
            results['group_delay_13_s'] = self.calculate_group_delay(df, freq, 1, 3)
            results['group_delay_23_s'] = self.calculate_group_delay(df, freq, 2, 3)

            # 5. Minimum insertion loss (Port 1-3 and Port 2-3)
            results['min_insertion_loss_13_dB'] = self.calculate_insertion_loss(df, 1, 3)
            results['min_insertion_loss_23_dB'] = self.calculate_insertion_loss(df, 2, 3)

            return results

        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return None


def batch_analyze_key_metrics(input_folder, output_folder):
    """Batch analyze key metrics"""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    analyzer = SParameterAnalyzer()
    all_results = []

    for i, file_path in enumerate(csv_files[:1000]):
        if i % 100 == 0:
            print(f"Processing file {i}...")

        results = analyzer.analyze_file(file_path)
        if results is not None:
            all_results.append(results)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.replace([np.inf, -np.inf], np.nan)

        # Save results
        output_csv = os.path.join(output_folder, "key_metrics_summary.csv")
        summary_df.to_csv(output_csv, index=False)
        print(f"Key metrics summary saved to: {output_csv}")

        # Display statistical results
        print_key_metrics_statistics(summary_df)

        # Generate charts
        generate_key_metrics_charts(summary_df, output_folder)

        return summary_df
    else:
        print("No files were successfully analyzed")
        return None


def print_key_metrics_statistics(summary_df):
    """Print key metrics statistics"""
    print("\n" + "=" * 60)
    print("Key Metrics Statistical Summary")
    print("=" * 60)

    metrics = {
        'Q_port1': ('Quality Factor (Port 1)', '', 2),
        'Q_port2': ('Quality Factor (Port 2)', '', 2),
        'coupling_coefficient': ('Coupling Coefficient', '', 4),
        'bandwidth_13_Hz': ('Bandwidth (Port 1-3)', 'GHz', 2),
        'bandwidth_23_Hz': ('Bandwidth (Port 2-3)', 'GHz', 2),
        'group_delay_13_s': ('Group Delay (Port 1-3)', 'ps', 2),
        'group_delay_23_s': ('Group Delay (Port 2-3)', 'ps', 2),
        'min_insertion_loss_13_dB': ('Min Insertion Loss (Port 1-3)', 'dB', 2),
        'min_insertion_loss_23_dB': ('Min Insertion Loss (Port 2-3)', 'dB', 2)
    }

    for col, (name, unit, precision) in metrics.items():
        if col in summary_df.columns:
            data = summary_df[col].dropna()
            if len(data) > 0:
                mean_val = data.mean()
                std_val = data.std()
                min_val = data.min()
                max_val = data.max()

                # Unit conversion
                if unit == 'GHz':
                    mean_val /= 1e9
                    std_val /= 1e9
                    min_val /= 1e9
                    max_val /= 1e9
                elif unit == 'ps':
                    mean_val *= 1e12
                    std_val *= 1e12
                    min_val *= 1e12
                    max_val *= 1e12

                print(f"\n{name}:")
                print(f"  Mean: {mean_val:.{precision}f} {unit}")
                print(f"  Std: {std_val:.{precision}f} {unit}")
                print(f"  Range: [{min_val:.{precision}f}, {max_val:.{precision}f}] {unit}")


def generate_key_metrics_charts(summary_df, output_folder):
    """Generate key metrics charts"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Quality Factor Distribution
    if 'Q_port1' in summary_df.columns and 'Q_port2' in summary_df.columns:
        q1_data = summary_df['Q_port1'].dropna()
        q2_data = summary_df['Q_port2'].dropna()

        axes[0, 0].hist([q1_data, q2_data], bins=20, alpha=0.7, label=['Port 1', 'Port 2'])
        axes[0, 0].set_xlabel('Quality Factor')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Quality Factor Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Coupling Coefficient Distribution
    if 'coupling_coefficient' in summary_df.columns:
        k_data = summary_df['coupling_coefficient'].dropna()
        axes[0, 1].hist(k_data, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Coupling Coefficient')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Coupling Coefficient Distribution')
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Bandwidth Comparison
    if 'bandwidth_13_Hz' in summary_df.columns and 'bandwidth_23_Hz' in summary_df.columns:
        bw13_data = summary_df['bandwidth_13_Hz'].dropna() / 1e9  # Convert to GHz
        bw23_data = summary_df['bandwidth_23_Hz'].dropna() / 1e9

        bar_positions = np.arange(2)
        bar_width = 0.6

        axes[0, 2].bar(bar_positions[0], bw13_data.mean(), bar_width,
                       yerr=bw13_data.std(), alpha=0.7, label='Port 1-3')
        axes[0, 2].bar(bar_positions[1], bw23_data.mean(), bar_width,
                       yerr=bw23_data.std(), alpha=0.7, label='Port 2-3')
        axes[0, 2].set_xticks(bar_positions)
        axes[0, 2].set_xticklabels(['1-3', '2-3'])
        axes[0, 2].set_ylabel('Bandwidth (GHz)')
        axes[0, 2].set_title('Average Bandwidth Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # 4. Group Delay Comparison
    if 'group_delay_13_s' in summary_df.columns and 'group_delay_23_s' in summary_df.columns:
        gd13_data = summary_df['group_delay_13_s'].dropna() * 1e12  # Convert to ps
        gd23_data = summary_df['group_delay_23_s'].dropna() * 1e12

        bar_positions = np.arange(2)

        axes[1, 0].bar(bar_positions[0], gd13_data.mean(), bar_width,
                       yerr=gd13_data.std(), alpha=0.7, label='Port 1-3')
        axes[1, 0].bar(bar_positions[1], gd23_data.mean(), bar_width,
                       yerr=gd23_data.std(), alpha=0.7, label='Port 2-3')
        axes[1, 0].set_xticks(bar_positions)
        axes[1, 0].set_xticklabels(['1-3', '2-3'])
        axes[1, 0].set_ylabel('Group Delay (ps)')
        axes[1, 0].set_title('Average Group Delay Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 5. Insertion Loss Comparison
    if 'min_insertion_loss_13_dB' in summary_df.columns and 'min_insertion_loss_23_dB' in summary_df.columns:
        il13_data = summary_df['min_insertion_loss_13_dB'].dropna()
        il23_data = summary_df['min_insertion_loss_23_dB'].dropna()

        bar_positions = np.arange(2)

        axes[1, 1].bar(bar_positions[0], il13_data.mean(), bar_width,
                       yerr=il13_data.std(), alpha=0.7, label='Port 1-3')
        axes[1, 1].bar(bar_positions[1], il23_data.mean(), bar_width,
                       yerr=il23_data.std(), alpha=0.7, label='Port 2-3')
        axes[1, 1].set_xticks(bar_positions)
        axes[1, 1].set_xticklabels(['1-3', '2-3'])
        axes[1, 1].set_ylabel('Insertion Loss (dB)')
        axes[1, 1].set_title('Minimum Insertion Loss Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Quality Factor vs Coupling Coefficient Scatter Plot
    if 'Q_port1' in summary_df.columns and 'coupling_coefficient' in summary_df.columns:
        q_data = summary_df['Q_port1'].dropna()
        k_data = summary_df['coupling_coefficient'].dropna()

        min_len = min(len(q_data), len(k_data))
        if min_len > 0:
            axes[1, 2].scatter(k_data[:min_len], q_data[:min_len], alpha=0.6)
            axes[1, 2].set_xlabel('Coupling Coefficient')
            axes[1, 2].set_ylabel('Quality Factor (Port 1)')
            axes[1, 2].set_title('Quality Factor vs Coupling Coefficient')
            axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_folder, "key_metrics_charts.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nKey metrics charts saved to: {chart_path}")


# Run analysis
if __name__ == "__main__":
    input_folder = "output_data_port"
    output_folder = "key_metrics_results_2"

    print("Starting key metrics analysis...")
    results = batch_analyze_key_metrics(input_folder, output_folder)

    if results is not None:
        print(f"\nAnalysis completed! Processed {len(results)} files")
