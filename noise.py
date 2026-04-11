import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
import os


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class InductorTransformer(nn.Module):
    def __init__(self, input_dim=6, output_dim=None, d_model=128, nhead=8,
                 num_encoder_layers=4, dim_feedforward=256, dropout=0.1):
        super(InductorTransformer, self).__init__()

        self.d_model = d_model
        self.output_dim = output_dim

        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.output_network = nn.Sequential(
            nn.Linear(d_model, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        output = self.output_network(x)
        return output


def load_trained_model(model_path='best_inductor_transformer.pth', device='cpu'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    output_dim = checkpoint['target_scaler'].mean_.shape[0]

    model = InductorTransformer(
        input_dim=6,
        output_dim=output_dim,
        d_model=128, nhead=8, num_encoder_layers=4,
        dim_feedforward=256, dropout=0.1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    feature_scaler = checkpoint['feature_scaler']
    target_scaler = checkpoint['target_scaler']
    n_freq = checkpoint.get('n_freq', output_dim // (9 * 2))

    print(f"Model loaded from {model_path}")
    print(f"Output dimension: {output_dim}")
    print(f"Number of frequency points: {n_freq}")

    return model, feature_scaler, target_scaler, n_freq


def add_gaussian_noise(features, noise_level, param_ranges):
    noise_std = noise_level * param_ranges
    noise = np.random.normal(0, noise_std, features.shape)
    return features + noise


def evaluate_model(model, test_loader, target_scaler, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds_scaled = np.vstack(all_preds)
    targets_scaled = np.vstack(all_targets)

    preds = target_scaler.inverse_transform(preds_scaled)
    targets = target_scaler.inverse_transform(targets_scaled)

    sample_dims = min(100, targets.shape[1])
    r2_scores = []
    for i in range(sample_dims):
        r2 = r2_score(targets[:, i], preds[:, i])
        r2_scores.append(r2)

    return np.mean(r2_scores)


def load_test_data(data_dir="inductor_s_params_dataset", max_samples=2000):
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from datatrain_tansformer import parse_inductor_data, load_sparams_from_npz, prepare_training_data

    txt_file_path = os.path.join(data_dir, "merged_data.txt")
    structural_params = parse_inductor_data(txt_file_path)

    if len(structural_params) > max_samples:
        structural_params = structural_params[:max_samples]

    npz_path = os.path.join(data_dir, "s_parameters.npz")
    sparams_results, freq = load_sparams_from_npz(npz_path)

    features, targets, feature_names, target_names = prepare_training_data(
        structural_params, sparams_results, freq
    )

    return features, targets, freq


def transformer_robustness_test(model, feature_scaler, target_scaler, features, targets,
                                param_ranges, noise_levels=[0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
                                device='cpu'):
    results = {'r2': []}
    model = model.to(device)
    model.eval()

    features_scaled = feature_scaler.transform(features)

    split = int(0.8 * len(features_scaled))
    X_test = features_scaled[split:]
    y_test = targets[split:]

    print(f"Test samples: {len(X_test)}")

    for noise_level in noise_levels:
        print(f"Testing noise level: {noise_level * 100:.0f}%")

        X_test_noisy = add_gaussian_noise(X_test.copy(), noise_level, param_ranges)

        X_test_tensor = torch.FloatTensor(X_test_noisy)
        y_test_tensor = torch.FloatTensor(y_test)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        r2 = evaluate_model(model, test_loader, target_scaler, device)
        results['r2'].append(r2)
        print(f"  Transformer R2: {r2:.4f}")

    return results


def plot_transformer_robustness(results, noise_levels):
    fig, ax = plt.subplots(figsize=(8, 5))

    baseline_r2 = results['r2'][0]
    r2_degradation = [(baseline_r2 - v) / baseline_r2 * 100 for v in results['r2']]

    ax.plot(noise_levels, r2_degradation,
            color='blue', marker='o', linewidth=2, markersize=8, label='Transformer')

    ax.axvline(x=0.05, color='gray', linestyle='--', alpha=0.7, label='5% Noise')
    ax.axvline(x=0.10, color='black', linestyle='--', alpha=0.7, label='10% Noise')
    ax.axhline(y=2, color='blue', linestyle=':', alpha=0.5, label='2% Degradation')

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('R2 Degradation (%)', fontsize=12)
    ax.set_title('Transformer Robustness Under Different Noise Levels', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(noise_levels)
    ax.set_xticklabels([f'{nl * 100:.0f}%' for nl in noise_levels])

    plt.tight_layout()
    plt.savefig('noise_robustness_transformer.png', dpi=150)
    plt.show()


def print_transformer_robustness_summary(results, noise_levels):
    print("\n" + "=" * 60)
    print("Transformer Noise Robustness Summary")
    print("=" * 60)

    baseline_idx = 0
    high_noise_idx = noise_levels.index(0.10) if 0.10 in noise_levels else 4

    clean_r2 = results['r2'][baseline_idx]
    noisy_r2 = results['r2'][high_noise_idx] if high_noise_idx < len(results['r2']) else results['r2'][-1]
    degradation = (clean_r2 - noisy_r2) / clean_r2 * 100

    print(f"Clean data R2: {clean_r2:.4f}")
    print(f"10% Noise R2:  {noisy_r2:.4f}")
    print(f"Performance degradation: {degradation:.2f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Transformer Robustness Test (Noise Injection)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    MODEL_PATH = "best_inductor_transformer.pth"
    DATA_DIR = "inductor_s_params_dataset"

    model, feature_scaler, target_scaler, n_freq = load_trained_model(MODEL_PATH, device)

    features, targets, freq = load_test_data(DATA_DIR, max_samples=2000)

    param_ranges = np.array([4, 4, 7, 7, 80, 80])

    noise_levels = [0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]

    results = transformer_robustness_test(
        model, feature_scaler, target_scaler, features, targets,
        param_ranges, noise_levels, device
    )

    plot_transformer_robustness(results, noise_levels)
    print_transformer_robustness_summary(results, noise_levels)

