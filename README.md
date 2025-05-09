# Federated Learning for IoMT Anomaly Detection

This project demonstrates a federated learning approach for anomaly detection in IoT environments using an autoencoder neural network.

## 👨💻 **Developer**  
### **Ayan Bhaumik**  
<div align="center" style="margin:20px 0">

[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-000000?style=for-the-badge&logo=google-chrome&logoColor=white)](https://ayanbhaumik.in/)
[![LinkedIn](https://img.shields.io/badge/👔_LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayan-bhaumik/)
[![Email](https://img.shields.io/badge/📧_Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:connect@ayanbhaumik.in)
[![GitHub](https://img.shields.io/badge/🐙_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ayan-Bhaumik)

</div>

## Overview

The system implements:
- A federated learning framework with multiple clients and a central server
- An autoencoder model for unsupervised anomaly detection
- Evaluation metrics and visualizations for model performance

## Key Features

- **Federated Learning**: Trains models across distributed IoT devices without sharing raw data
- **Autoencoder Architecture**: Learns compressed representations of normal IoT device behavior
- **Anomaly Detection**: Identifies deviations from normal patterns using reconstruction error
- **Visual Analytics**: Includes loss distribution plots, anomaly score visualizations, and t-SNE projections

## Implementation Details

### Model Architecture
- Encoder: 20 → 12 → 6 dimensions
- Decoder: 6 → 12 → 20 dimensions
- Activation: ReLU in hidden layers, Sigmoid in output layer
- Loss: Mean Squared Error (MSE)

### Training Process
- Multiple clients train local models on their own data
- Server aggregates model weights using Federated Averaging
- Training occurs over multiple communication rounds

### Evaluation Metrics
- AUC-ROC score for anomaly detection performance
- Dynamic thresholding based on reconstruction loss percentiles

## Usage

1. Install dependencies:
```bash
pip install numpy torch scikit-learn matplotlib
```

2. Run the federated training:
```python
global_model = federated_training(num_clients=5, rounds=10)
```

3. Evaluate on test data:
```python
X_test, y_test = generate_iot_data(1000, anomaly_ratio=0.1)
anomalies, losses, threshold = detect_anomalies(global_model, X_test)
```

4. Visualize results:
```python
plot_loss_distribution(losses, threshold)
plot_anomaly_scatter(losses, y_test, threshold)
plot_tsne(encoded, y_test)
```

## Results

The model outputs:
- Reconstruction loss distribution plot
- Anomaly score visualization
- t-SNE projection of encoded features
- AUC-ROC score for detection performance

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

## Future Enhancements

- Add differential privacy for enhanced security
- Implement adaptive thresholding
- Extend to semi-supervised learning
- Add real-time detection capabilities

## License

This project is open-source and available under the MIT License.

## 📬 **Contact**  

<div align="center">

[![Email](https://img.shields.io/badge/📨_connect@ayanbhaumik.in-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:connect@ayanbhaumik.in)
[![LinkedIn](https://img.shields.io/badge/👔_Let's_Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayan-bhaumik/)

</div>
