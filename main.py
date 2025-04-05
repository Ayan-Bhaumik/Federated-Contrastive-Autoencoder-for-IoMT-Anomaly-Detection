!pip install -q numpy torch scikit-learn matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_iot_data(n_samples=1000, anomaly_ratio=0.05):
    X, y = make_classification(n_samples=n_samples, n_features=20, n_informative=15,
                               n_clusters_per_class=1, weights=[1 - anomaly_ratio])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = np.where(y == 0, 0, 1)
    return X.astype(np.float32), y

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(model, data, epochs=5, lr=0.01):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = torch.tensor(data).to(device)
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
    return model.state_dict()

def federated_avg(models):
    avg_model = {}
    for k in models[0].keys():
        avg_model[k] = sum([model[k] for model in models]) / len(models)
    return avg_model

def federated_training(num_clients=5, rounds=10):
    input_dim = 20
    global_model = Autoencoder(input_dim).to(device)
    clients_data = [generate_iot_data(500)[0] for _ in range(num_clients)]
    for r in range(rounds):
        client_models = []
        for client_data in clients_data:
            local_model = Autoencoder(input_dim).to(device)
            local_model.load_state_dict(global_model.state_dict())
            client_state = train_autoencoder(local_model, client_data)
            client_models.append(client_state)
        global_weights = federated_avg(client_models)
        global_model.load_state_dict(global_weights)
        print(f"Round {r+1} complete.")
    return global_model

def detect_anomalies(model, test_data, threshold=None):
    model.eval()
    test_tensor = torch.tensor(test_data).to(device)
    with torch.no_grad():
        reconstructed = model(test_tensor).cpu().numpy()
    loss = np.mean((reconstructed - test_data) ** 2, axis=1)
    if threshold is None:
        threshold = np.percentile(loss, 95)
    anomalies = loss > threshold
    return anomalies, loss, threshold

def plot_loss_distribution(losses, threshold):
    plt.figure(figsize=(10, 4))
    plt.hist(losses, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.title('Reconstruction Loss Distribution')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_anomaly_scatter(losses, y_true, threshold):
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(losses)), losses, c=y_true, cmap='coolwarm', s=10)
    plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title("Anomaly Score (Loss) vs Index")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tsne(encoded, y_true):
    tsne = TSNE(n_components=2, random_state=42)
    z = tsne.fit_transform(encoded)
    plt.figure(figsize=(6, 6))
    plt.scatter(z[:, 0], z[:, 1], c=y_true, cmap='coolwarm', s=15)
    plt.title("t-SNE of Encoded Features")
    plt.grid(True)
    plt.show()

global_model = federated_training()
X_test, y_test = generate_iot_data(1000, anomaly_ratio=0.1)
anomalies, losses, threshold = detect_anomalies(global_model, X_test)
print("AUC Score:", roc_auc_score(y_test, losses))
plot_loss_distribution(losses, threshold)
plot_anomaly_scatter(losses, y_test, threshold)
encoded = global_model.encoder(torch.tensor(X_test).to(device)).cpu().detach().numpy()
plot_tsne(encoded, y_test)
