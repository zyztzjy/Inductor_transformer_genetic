import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import re
import warnings
import glob
import os
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from scipy import signal
import time

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def parse_inductor_data(txt_file_path):

    if not os.path.exists(txt_file_path):
        print(f"File not found: {txt_file_path}")
        return []

    with open(txt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    patterns = [
        r'turns_top=([\d.]+),\s*turns_bot=([\d.]+),\s*linewidth_top=([\d.]+),\s*linewidth_bot=([\d.]+),\s*center_gap=([\d.]+),\s*inner_diam=([\d.]+)',
        r'N_t[=:]\s*([\d.]+).*?N_b[=:]\s*([\d.]+).*?W_t[=:]\s*([\d.]+).*?W_b[=:]\s*([\d.]+).*?G_c[=:]\s*([\d.]+).*?D_i[=:]\s*([\d.]+)',
        r'(\d+),\s*(\d+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)'
    ]

    structural_params = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    params = {
                        'N_t': int(float(match[0])),
                        'N_b': int(float(match[1])),
                        'W_t': float(match[2]),
                        'W_b': float(match[3]),
                        'G_c': float(match[4]),
                        'D_i': float(match[5])
                    }
                    if (1 <= params['N_t'] <= 5 and 1 <= params['N_b'] <= 5 and
                            3 <= params['W_t'] <= 10 and 3 <= params['W_b'] <= 10 and
                            40 <= params['G_c'] <= 120 and 20 <= params['D_i'] <= 100):
                        structural_params.append(params)
                except (ValueError, IndexError):
                    continue
            if structural_params:
                break

    print(f"Successfully parsed {len(structural_params)} inductor structures")

    unique_params = []
    seen = set()
    for p in structural_params:
        key = (p['N_t'], p['N_b'], p['W_t'], p['W_b'], p['G_c'], p['D_i'])
        if key not in seen:
            seen.add(key)
            unique_params.append(p)

    print(f"Remaining after deduplication: {len(unique_params)} unique structures")

    if len(unique_params) > 0:
        print("\nFirst 5 samples:")
        for i in range(min(5, len(unique_params))):
            print(f"  Sample{i + 1}: N_t={unique_params[i]['N_t']}, N_b={unique_params[i]['N_b']}, "
                  f"W_t={unique_params[i]['W_t']:.2f}, W_b={unique_params[i]['W_b']:.2f}, "
                  f"G_c={unique_params[i]['G_c']:.2f}, D_i={unique_params[i]['D_i']:.2f}")

    return unique_params


def load_sparams_from_npz(npz_path):

    if not os.path.exists(npz_path):
        print(f"NPZ file not found: {npz_path}")
        return None, None

    with np.load(npz_path, allow_pickle=True) as data:
        freq = data['freq']
        n_samples = data['S11_mag'].shape[0]

        print(f"Loading NPZ file: {npz_path}")
        print(f"  Number of samples: {n_samples}")
        print(f"  Number of frequency points: {len(freq)}")
        print(f"  Frequency range: {freq[0]:.1f} - {freq[-1]:.1f} GHz")

        sparams_results = []
        s_params = ['S11', 'S12', 'S13', 'S21', 'S22', 'S23', 'S31', 'S32', 'S33']

        for i in range(n_samples):
            result = {}
            for sp in s_params:
                result[f'{sp}_mag'] = data[f'{sp}_mag'][i].copy()
                result[f'{sp}_phase'] = data[f'{sp}_phase'][i].copy()
            sparams_results.append(result)

        print(f"Successfully loaded {len(sparams_results)} S-parameter results")
        return sparams_results, freq


def log_freq_normalization(freq, f_min=0.1, f_max=120):

    freq_log = np.log10(freq)
    f_min_log = np.log10(f_min)
    f_max_log = np.log10(f_max)
    freq_norm = (freq_log - f_min_log) / (f_max_log - f_min_log)
    return freq_norm


def remove_outliers_3sigma(data, n_sigma=3):

    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - n_sigma * std
    upper_bound = mean + n_sigma * std
    return data[(data >= lower_bound) & (data <= upper_bound)]


def prepare_training_data(structural_params, sparams_results, freq):

    min_len = min(len(structural_params), len(sparams_results))
    structural_params = structural_params[:min_len]
    sparams_results = sparams_results[:min_len]

    print(f"\nUsing {min_len} samples for training")

    features_array = np.array([
        [params['N_t'], params['N_b'], params['W_t'], params['W_b'], params['G_c'], params['D_i']]
        for params in structural_params
    ], dtype=np.float32)

    s_params = ['S11', 'S12', 'S13', 'S21', 'S22', 'S23', 'S31', 'S32', 'S33']
    n_freq = len(freq)
    n_samples = len(sparams_results)

    targets_array = np.zeros((n_samples, len(s_params) * 2 * n_freq), dtype=np.float32)

    print(f"Processing S-parameters for {n_samples} samples...")

    batch_size = 1000
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        for j in range(i, end_idx):
            result = sparams_results[j]
            idx = 0
            for sp in s_params:
                mag_array = result[f'{sp}_mag']
                phase_array = result[f'{sp}_phase']
                targets_array[j, idx::2] = mag_array
                targets_array[j, idx + 1::2] = phase_array
                idx += 2 * n_freq

        print(f"  Progress: {min(end_idx, n_samples)}/{n_samples} ({min(end_idx, n_samples) / n_samples * 100:.1f}%)")

    print("\nPerforming outlier removal (3-sigma criterion)...")
    valid_indices = []
    for i in range(targets_array.shape[1]):
        col_data = targets_array[:, i]
        if np.all(col_data == 0):
            continue
        filtered = remove_outliers_3sigma(col_data, n_sigma=3)
        valid_mask = np.isin(col_data, filtered)
        if i == 0:
            combined_mask = valid_mask
        else:
            combined_mask = combined_mask & valid_mask

    features_array = features_array[combined_mask]
    targets_array = targets_array[combined_mask]

    removed_count = n_samples - len(features_array)
    removal_rate = removed_count / n_samples * 100
    print(f"Removed outliers: {removed_count} ({removal_rate:.2f}%)")
    print(f"Valid samples remaining: {len(features_array)}")

    target_names = []
    for sp in s_params:
        for f_idx in range(n_freq):
            target_names.append(f'{sp}_mag_f{f_idx:03d}')
            target_names.append(f'{sp}_phase_f{f_idx:03d}')

    nan_count = np.sum(np.isnan(targets_array))
    if nan_count > 0:
        print(f"Warning: NaN values found, count={nan_count}")
        targets_array = np.nan_to_num(targets_array, nan=0)

    inf_count = np.sum(np.isinf(targets_array))
    if inf_count > 0:
        print(f"Warning: Inf values found, count={inf_count}")
        targets_array = np.nan_to_num(targets_array, posinf=0, neginf=0)

    print(f"Feature dimension: {features_array.shape}")
    print(f"Target dimension: {targets_array.shape}")
    print(f"Number of targets per sample: {targets_array.shape[1]}")

    return features_array, targets_array, ['N_t', 'N_b', 'W_t', 'W_b', 'G_c', 'D_i'], target_names


class InductorDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


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


class WeightedMSELoss(nn.Module):


    def __init__(self, s_param_weights=None):
        super(WeightedMSELoss, self).__init__()
        if s_param_weights is None:
            self.s_param_weights = {
                'S11': 1.2, 'S22': 1.2, 'S33': 1.2,
                'S12': 1.0, 'S21': 1.0, 'S13': 1.0, 'S31': 1.0,
                'S23': 0.8, 'S32': 0.8
            }
        else:
            self.s_param_weights = s_param_weights

    def forward(self, pred, target, n_freq):
        batch_size = pred.shape[0]
        n_sparams = 9

        sparam_dim = 2 * n_freq
        total_loss = 0.0

        for i, sp in enumerate(['S11', 'S12', 'S13', 'S21', 'S22', 'S23', 'S31', 'S32', 'S33']):
            start_idx = i * sparam_dim
            end_idx = (i + 1) * sparam_dim

            pred_sp = pred[:, start_idx:end_idx]
            target_sp = target[:, start_idx:end_idx]

            weight = self.s_param_weights[sp]
            loss_sp = F.mse_loss(pred_sp, target_sp, reduction='mean')
            total_loss += weight * loss_sp

        return total_loss / n_sparams


class CombinedLoss(nn.Module):


    def __init__(self, s_param_weights=None, delta=1.0, lambda_l2=1e-4):
        super(CombinedLoss, self).__init__()
        self.weighted_mse = WeightedMSELoss(s_param_weights)
        self.delta = delta
        self.lambda_l2 = lambda_l2

    def forward(self, pred, target, n_freq, model=None):
        batch_size = pred.shape[0]
        n_sparams = 9
        sparam_dim = 2 * n_freq

        weighted_mse_loss = 0.0
        huber_loss = 0.0

        for i, sp in enumerate(['S11', 'S12', 'S13', 'S21', 'S22', 'S23', 'S31', 'S32', 'S33']):
            start_idx = i * sparam_dim
            end_idx = (i + 1) * sparam_dim

            pred_sp = pred[:, start_idx:end_idx]
            target_sp = target[:, start_idx:end_idx]

            weight = self.weighted_mse.s_param_weights[sp]
            weighted_mse_loss += weight * F.mse_loss(pred_sp, target_sp, reduction='mean')
            huber_loss += F.huber_loss(pred_sp, target_sp, delta=self.delta, reduction='mean')

        weighted_mse_loss = weighted_mse_loss / n_sparams
        huber_loss = huber_loss / n_sparams

        total_loss = weighted_mse_loss + huber_loss

        if model is not None:
            l2_reg = sum(p.norm(2) for p in model.parameters())
            total_loss = total_loss + self.lambda_l2 * l2_reg

        return total_loss


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


def mixup_augmentation(x, y, alpha=0.2):
    """MixUp data augmentation"""
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(batch_size)
    x_mixed = lam * x + (1 - lam) * x[indices]
    y_mixed = lam * y + (1 - lam) * y[indices]
    return x_mixed, y_mixed


def add_gaussian_noise(features, noise_std=0.01):
    """Add Gaussian noise for data augmentation"""
    noise = np.random.normal(0, noise_std, features.shape)
    return features + noise


def random_scaling(features, scale_range=(0.95, 1.05)):
    """Random scaling augmentation"""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return features * scale


def kfold_cross_validation(features, targets, n_folds=5, epochs_per_fold=100):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        print(f"\n{'=' * 40}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'=' * 40}")

        X_train_fold = features[train_idx]
        X_val_fold = features[val_idx]
        y_train_fold = targets[train_idx]
        y_val_fold = targets[val_idx]

        model = InductorTransformer(
            input_dim=features.shape[1],
            output_dim=targets.shape[1],
            d_model=128, nhead=8, num_encoder_layers=4,
            dim_feedforward=256, dropout=0.1
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=30)
        criterion = CombinedLoss(delta=1.0, lambda_l2=1e-4)

        n_freq = targets.shape[1] // (9 * 2)

        train_dataset = InductorDataset(X_train_fold, y_train_fold)
        val_dataset = InductorDataset(X_val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs_per_fold):
            model.train()
            train_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                if np.random.random() < 0.3:
                    batch_features, batch_targets = mixup_augmentation(batch_features, batch_targets, alpha=0.2)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets, n_freq, model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    outputs = model(batch_features)
                    val_loss += criterion(outputs, batch_targets, n_freq).item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch [{epoch + 1:4d}/{epochs_per_fold}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if patience_counter >= 50:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        fold_results.append(best_val_loss)
        print(f"Fold {fold + 1} best validation loss: {best_val_loss:.6f}")

    print(f"\n5-fold CV results: mean loss = {np.mean(fold_results):.6f}, std = {np.std(fold_results):.6f}")
    return fold_results


def train_inductor_transformer(data_dir="inductor_s_params_dataset", epochs=1200):

    print("=" * 60)
    print("Inductor-Transformer Model Training")
    print("=" * 60)

    print("\nStep 1: Loading inductor structure data...")
    txt_file_path = os.path.join(data_dir, "merged_data.txt")
    structural_params = parse_inductor_data(txt_file_path)

    MAX_SAMPLES = 10400
    if len(structural_params) > MAX_SAMPLES:
        print(f"Warning: Too many samples, limiting to {MAX_SAMPLES}")
        structural_params = structural_params[:MAX_SAMPLES]
    if len(structural_params) == 0:
        print("Error: No geometric parameter data found!")
        return None, None, None, None, None

    print("\nStep 2: Loading S-parameter results...")
    npz_path = os.path.join(data_dir, "s_parameters.npz")
    sparams_results, freq = load_sparams_from_npz(npz_path)

    if sparams_results is None:
        print("Error: No S-parameter data found!")
        return None, None, None, None, None

    print("\nStep 3: Preparing training data...")
    features, targets, feature_names, target_names = prepare_training_data(
        structural_params, sparams_results, freq
    )

    if len(features) == 0:
        print("Error: Training data is empty!")
        return None, None, None, None, None

    print("\nStep 4: Data preprocessing and normalization...")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets)

    print("\nStep 5: Data augmentation...")
    features_aug = add_gaussian_noise(features_scaled, noise_std=0.01)
    features_aug2 = random_scaling(features_scaled, scale_range=(0.95, 1.05))

    features_scaled = np.vstack([features_scaled, features_aug, features_aug2])
    targets_scaled = np.vstack([targets_scaled, targets_scaled, targets_scaled])

    idx = np.random.permutation(len(features_scaled))
    features_scaled = features_scaled[idx]
    targets_scaled = targets_scaled[idx]
    print(f"Augmented dataset size: {len(features_scaled)}")

    print("\nStep 6: Performing 5-fold cross validation...")
    n_freq = targets.shape[1] // (9 * 2)
    cv_results = kfold_cross_validation(features_scaled, targets_scaled, n_folds=5, epochs_per_fold=100)

    print("\nStep 7: Training final model on full dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_scaled, targets_scaled, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    train_loader = DataLoader(InductorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(InductorDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(InductorDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = InductorTransformer(
        input_dim=features.shape[1],
        output_dim=targets.shape[1],
        d_model=128, nhead=8, num_encoder_layers=4,
        dim_feedforward=256, dropout=0.1
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output dimension: {targets.shape[1]}")
    print(f"Input dimension: {features.shape[1]} (6 geometric parameters: N_t, N_b, W_t, W_b, G_c, D_i)")

    criterion = CombinedLoss(delta=1.0, lambda_l2=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=30)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    patience = 150

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            if np.random.random() < 0.3:
                batch_features, batch_targets = mixup_augmentation(batch_features, batch_targets, alpha=0.2)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets, n_freq, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_targets, n_freq).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1:4d}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        if early_stop_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest validation loss: {best_val_loss:.6f}")

    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.numpy())

    preds_scaled = np.vstack(all_preds)
    targets_scaled = np.vstack(all_targets)

    preds_original = target_scaler.inverse_transform(preds_scaled)
    targets_original = target_scaler.inverse_transform(targets_scaled)

    sample_dims = min(100, targets_original.shape[1])
    r2_scores = []
    mape_scores = []

    for i in range(sample_dims):
        r2 = r2_score(targets_original[:, i], preds_original[:, i])
        mape = np.mean(
            np.abs((targets_original[:, i] - preds_original[:, i]) / (np.abs(targets_original[:, i]) + 1e-8))) * 100
        r2_scores.append(r2)
        mape_scores.append(mape)

    avg_r2 = np.mean(r2_scores)
    avg_mape = np.mean(mape_scores)

    print(f"\nAverage R² (first {sample_dims} dims): {avg_r2:.4f}")
    print(f"Average MAPE (first {sample_dims} dims): {avg_mape:.2f}%")

    print("\nStep 8: Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': feature_names,
        'target_names': target_names,
        'best_val_loss': best_val_loss,
        'avg_r2': avg_r2,
        'avg_mape': avg_mape,
        'n_freq': n_freq
    }, 'best_inductor_transformer.pth')

    print(f"\nModel saved to: best_inductor_transformer.pth")

    print("\nStep 9: Generating visualizations...")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.bar(range(len(r2_scores[:50])), r2_scores[:50])
    plt.xlabel('Target Index')
    plt.ylabel('R² Score')
    plt.title(f'R² per Target (Avg: {avg_r2:.3f})')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.bar(range(len(mape_scores[:50])), mape_scores[:50])
    plt.xlabel('Target Index')
    plt.ylabel('MAPE (%)')
    plt.title(f'MAPE per Target (Avg: {avg_mape:.2f}%)')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    sample_idx = np.random.randint(0, len(preds_original), min(5, len(preds_original)))
    for idx in sample_idx:
        plt.scatter(targets_original[idx, :100], preds_original[idx, :100], alpha=0.5, s=1)
    plt.plot([targets_original[:, :100].min(), targets_original[:, :100].max()],
             [targets_original[:, :100].min(), targets_original[:, :100].max()],
             'r--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.title('Prediction vs Truth')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()

    return model, feature_scaler, target_scaler, feature_names, target_names


class InductorPredictor:


    def __init__(self, model_path='best_inductor_transformer.pth'):
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = None
        self.target_names = None
        self.n_freq = None
        self.load_model()

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=device)

        output_dim = checkpoint['target_scaler'].mean_.shape[0]

        self.model = InductorTransformer(
            input_dim=6,
            output_dim=output_dim,
            d_model=128, nhead=8, num_encoder_layers=4,
            dim_feedforward=256, dropout=0.1
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        self.feature_names = checkpoint['feature_names']
        self.target_names = checkpoint['target_names']
        self.n_freq = checkpoint.get('n_freq', output_dim // (9 * 2))

        print(f"Model loaded from {self.model_path}")
        print(f"Output dimension: {output_dim}")
        print(f"Number of frequency points: {self.n_freq}")

    def predict(self, geometric_params):
        """
        Predict S-parameters from geometric parameters

        Args:
            geometric_params: dict or list/tuple of 6 values [N_t, N_b, W_t, W_b, G_c, D_i]

        Returns:
            S-parameters as numpy array
        """
        if isinstance(geometric_params, dict):
            features = np.array([[
                geometric_params['N_t'],
                geometric_params['N_b'],
                geometric_params['W_t'],
                geometric_params['W_b'],
                geometric_params['G_c'],
                geometric_params['D_i']
            ]], dtype=np.float32)
        else:
            features = np.array([geometric_params], dtype=np.float32)

        features_scaled = self.feature_scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(device)

        with torch.no_grad():
            pred_scaled = self.model(features_tensor).cpu().numpy()

        pred_original = self.target_scaler.inverse_transform(pred_scaled)

        return pred_original[0]

    def get_sparams_dict(self, geometric_params, freq=None):

        pred = self.predict(geometric_params)

        s_params = ['S11', 'S12', 'S13', 'S21', 'S22', 'S23', 'S31', 'S32', 'S33']
        sparam_dim = 2 * self.n_freq

        result = {}
        for i, sp in enumerate(s_params):
            start_idx = i * sparam_dim
            mag = pred[start_idx::2]
            phase = pred[start_idx + 1::2]
            result[f'{sp}_mag'] = mag
            result[f'{sp}_phase'] = phase

        if freq is not None:
            result['freq'] = freq[:len(mag)]

        return result


if __name__ == "__main__":
    DATA_DIR = "inductor_s_params_dataset"

    model, feature_scaler, target_scaler, feature_names, target_names = \
        train_inductor_transformer(DATA_DIR, epochs=1200)

    print("\nTraining completed!")

    predictor = InductorPredictor('best_inductor_transformer.pth')

    test_params = {
        'N_t': 2, 'N_b': 1, 'W_t': 5.2, 'W_b': 5.8, 'G_c': 68, 'D_i': 39
    }

    print("\nTesting predictor with sample parameters:")
    print(f"Input: {test_params}")
    pred_sparams = predictor.predict(test_params)
    print(f"Output S-parameter vector shape: {pred_sparams.shape}")