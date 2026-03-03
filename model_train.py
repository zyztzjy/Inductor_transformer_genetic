import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import re
import warnings
import glob
import os

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def parse_inductor_data(txt_file_path):
    """Parse inductor data from txt file"""
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = r'turns_top=(\d+),\s*turns_bot=(\d+),\s*linewidth_top=([\d.]+),\s*linewidth_bot=([\d.]+),\s*center_gap=([\d.]+),\s*inner_diam=([\d.]+)'

    matches = re.findall(pattern, content)

    structural_params = []
    for match in matches:
        params = {
            'turns_top': int(match[0]),
            'turns_bot': int(match[1]),
            'linewidth_top': float(match[2]),
            'linewidth_bot': float(match[3]),
            'center_gap': float(match[4]),
            'inner_diam': float(match[5])
        }
        structural_params.append(params)

    print(f"Successfully parsed {len(structural_params)} inductor structures")

    return structural_params


def load_sparams_results(csv_folder):
    """Load S-parameter analysis results"""
    csv_files = glob.glob(os.path.join(csv_folder, "*key_metrics*.csv"))

    if not csv_files:
        raise FileNotFoundError("No S-parameter result files found")

    latest_file = max(csv_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)

    sparams_results = []
    for _, row in df.iterrows():
        result = {
            'Q_port1': row.get('Q_port1', np.nan),
            'Q_port2': row.get('Q_port2', np.nan),
            'coupling_coefficient': row.get('coupling_coefficient', np.nan),
            'bandwidth_13_Hz': row.get('bandwidth_13_Hz', np.nan),
            'bandwidth_23_Hz': row.get('bandwidth_23_Hz', np.nan),
            'group_delay_13_s': row.get('group_delay_13_s', np.nan),
            'group_delay_23_s': row.get('group_delay_23_s', np.nan),
            'min_insertion_loss_13_dB': row.get('min_insertion_loss_13_dB', np.nan),
            'min_insertion_loss_23_dB': row.get('min_insertion_loss_23_dB', np.nan)
        }
        sparams_results.append(result)

    print(f"Successfully loaded {len(sparams_results)} S-parameter results")

    return sparams_results


class InductorDataset(Dataset):
    """Inductor dataset"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class EnhancedInductorTransformer(nn.Module):
    """Enhanced Transformer model for inductors"""
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=256, dropout=0.1):
        super(EnhancedInductorTransformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_bn = nn.BatchNorm1d(input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layers = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_projection(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        encoded = self.transformer_encoder(x)

        if encoded.size(1) > 1:
            encoded = encoded.mean(dim=1)
        else:
            encoded = encoded.squeeze(1)

        output = self.output_layers(encoded)
        return output


def augment_data(features, targets, augmentation_factor=0.1):
    """Data augmentation by adding noise and scaling"""
    augmented_features = features.copy()
    augmented_targets = targets.copy()

    noise_features = np.random.normal(0, augmentation_factor, features.shape)
    noise_targets = np.random.normal(0, augmentation_factor, targets.shape)

    augmented_features = np.vstack([augmented_features, features + noise_features])
    augmented_targets = np.vstack([augmented_targets, targets + noise_targets])

    scale_factors = np.random.uniform(0.95, 1.05, (features.shape[0], 1))
    scaled_features = features * scale_factors
    scaled_targets = targets * scale_factors

    augmented_features = np.vstack([augmented_features, scaled_features])
    augmented_targets = np.vstack([augmented_targets, scaled_targets])

    return augmented_features, augmented_targets


def prepare_training_data(structural_params, sparams_results):
    """Prepare training data"""
    min_len = min(len(structural_params), len(sparams_results))
    structural_params = structural_params[:min_len]
    sparams_results = sparams_results[:min_len]

    print(f"Using {min_len} samples for training")

    structural_features = []
    feature_names = [
        'turns_top', 'linewidth_top', 'linewidth_bot',
        'center_gap', 'inner_diam', 'total_turns', 'width_ratio', 'size_ratio'
    ]

    for params in structural_params:
        features = [
            params['turns_top'],
            params['linewidth_top'],
            params['linewidth_bot'],
            params['center_gap'],
            params['inner_diam'],
            params['turns_top'],
            params['linewidth_top'] / max(params['linewidth_bot'], 1e-6),
            params['inner_diam'] / max(params['center_gap'], 1e-6),
        ]
        structural_features.append(features)

    target_columns = [
        'Q_port1', 'Q_port2', 'coupling_coefficient',
        'bandwidth_13_Hz', 'bandwidth_23_Hz',
        'group_delay_13_s', 'group_delay_23_s',
        'min_insertion_loss_13_dB', 'min_insertion_loss_23_dB'
    ]

    targets = []
    for result in sparams_results:
        target_values = [result[col] for col in target_columns]
        targets.append(target_values)

    features_array = np.array(structural_features)
    targets_array = np.array(targets)

    valid_indices = ~np.any(np.isnan(targets_array), axis=1)
    features_array = features_array[valid_indices]
    targets_array = targets_array[valid_indices]

    print(f"Remaining valid samples after cleaning: {len(features_array)}")

    return features_array, targets_array, feature_names, target_columns


def train_inductor_transformer(txt_file_path, sparams_results_folder, epochs=800):
    """Training function with improved early stopping"""
    print("Step 1: Parsing inductor structure data...")
    structural_params = parse_inductor_data(txt_file_path)

    print("Step 2: Loading S-parameter results...")
    sparams_results = load_sparams_results(sparams_results_folder)

    print("Step 3: Preparing training data...")
    features, targets, feature_names, target_names = prepare_training_data(
        structural_params, sparams_results
    )

    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets)

    print("Step 4: Data augmentation...")
    features_scaled, targets_scaled = augment_data(features_scaled, targets_scaled, augmentation_factor=0.1)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, targets_scaled, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    train_dataset = InductorDataset(X_train, y_train)
    val_dataset = InductorDataset(X_val, y_val)
    test_dataset = InductorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = features.shape[1]
    output_dim = targets.shape[1]

    model = EnhancedInductorTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.7)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    patience = 150
    early_stopping_counter = 0

    print("Starting training of enhanced Transformer model...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Final test loss: {test_loss:.6f}')
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': feature_names,
        'target_names': target_names,
        'best_val_loss': best_val_loss,
        'epoch': epochs
    }, 'best_inductor_transformer.pth')

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    feature_importance = np.abs(model.input_projection.weight.data.cpu().numpy()).mean(axis=0)
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Analysis')

    plt.tight_layout()
    plt.savefig('enhanced_transformer_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, feature_scaler, target_scaler, feature_names, target_names


def cross_validation_train(txt_file_path, sparams_results_folder, k_folds=5):
    """K-fold cross validation training"""
    structural_params = parse_inductor_data(txt_file_path)
    sparams_results = load_sparams_results(sparams_results_folder)
    features, targets, feature_names, target_names = prepare_training_data(
        structural_params, sparams_results
    )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets)

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features_scaled)):
        print(f"Training fold {fold + 1}/{k_folds}...")

        X_train_fold, X_val_fold = features_scaled[train_idx], features_scaled[val_idx]
        y_train_fold, y_val_fold = targets_scaled[train_idx], targets_scaled[val_idx]

        train_dataset = InductorDataset(X_train_fold, y_train_fold)
        val_dataset = InductorDataset(X_val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        input_dim = features_scaled.shape[1]
        output_dim = targets_scaled.shape[1]
        model = EnhancedInductorTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.7)

        best_val_loss = float('inf')
        patience = 100
        early_stopping_counter = 0

        for epoch in range(300):
            model.train()
            train_loss = 0.0

            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)

                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                break

        fold_results.append(best_val_loss)
        print(f"Fold {fold + 1} validation loss: {best_val_loss:.6f}")

    print(f"{k_folds}-fold cross validation average loss: {np.mean(fold_results):.6f} ± {np.std(fold_results):.6f}")
    return fold_results


class InductorPredictor:
    """Inductor performance predictor"""
    def __init__(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=device)

        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        self.feature_names = checkpoint['feature_names']
        self.target_names = checkpoint['target_names']

        input_dim = len(self.feature_names)
        output_dim = len(self.target_names)

        self.model = EnhancedInductorTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully! Validation loss: {checkpoint.get('best_val_loss', 'N/A')}")

    def predict(self, structural_params):
        """Predict inductor performance"""
        features = []
        for params in structural_params:
            feature_vec = [
                params['turns_top'],
                params['linewidth_top'],
                params['linewidth_bot'],
                params['center_gap'],
                params['inner_diam'],
                params['turns_top'],
                params['linewidth_top'] / max(params['linewidth_bot'], 1e-6),
                params['inner_diam'] / max(params['center_gap'], 1e-6),
            ]
            features.append(feature_vec)

        features = np.array(features)
        features_scaled = self.feature_scaler.transform(features)

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            predictions_scaled = self.model(features_tensor).cpu().numpy()

        predictions = self.target_scaler.inverse_transform(predictions_scaled)

        results = []
        for pred in predictions:
            result_dict = {}
            for i, target_name in enumerate(self.target_names):
                result_dict[target_name] = pred[i]
            results.append(result_dict)

        return results


if __name__ == "__main__":
    txt_file_path = "merged_data.txt"
    sparams_results_folder = "key_metrics_results"

    print("Starting training of enhanced inductor Transformer model...")
    model, feature_scaler, target_scaler, feature_names, target_names = train_inductor_transformer(
        txt_file_path, sparams_results_folder, epochs=1200
    )

    print("\nTraining completed!")
    print(f"Features: {feature_names}")
    print(f"Targets: {target_names}")

    predictor = InductorPredictor('best_inductor_transformer.pth')

    new_designs = [
        {
            'turns_top': 2, 'turns_bot': 1,
            'linewidth_top': 7, 'linewidth_bot': 8,
            'center_gap': 69, 'inner_diam': 63
        },
        {
            'turns_top': 3, 'turns_bot': 2,
            'linewidth_top': 8, 'linewidth_bot': 9,
            'center_gap': 70, 'inner_diam': 65
        }
    ]

    predictions = predictor.predict(new_designs)

    print("\nPrediction results:")
    for i, pred in enumerate(predictions):
        print(f"\nDesign {i + 1}:")
        for key, value in pred.items():
            if 'bandwidth' in key:
                print(f"  {key}: {value / 1e9:.2f} GHz")
            elif 'group_delay' in key:
                print(f"  {key}: {value * 1e12:.2f} ps")
            else:
                print(f"  {key}: {value:.4f}")
