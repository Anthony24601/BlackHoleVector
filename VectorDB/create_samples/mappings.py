import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

output_dir = "nonlin_data_test"
os.makedirs(output_dir, exist_ok=True)

def generate_clusters(n_samples_per_cluster, n_features):
    cluster_0 = np.random.randn(n_samples_per_cluster, n_features) + 2
    cluster_1 = np.random.randn(n_samples_per_cluster, n_features) - 2
    return cluster_0, cluster_1

def linear_mapping(low_dim_data, high_dim_size):
    n_samples, low_dim_size = low_dim_data.shape
    random_matrix = np.random.randn(low_dim_size, high_dim_size) # Maybe not random. Try not that
    return np.dot(low_dim_data, random_matrix)

class NonLinearMapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonLinearMapping, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def nonlinear_mapping(low_dim_data, model):
    low_dim_data = torch.tensor(low_dim_data, dtype=torch.float32)
    with torch.no_grad():
        high_dim_data = model(low_dim_data).numpy()
    return high_dim_data

def save_images(data, image_size, prefix, cluster_label):
    scaler = MinMaxScaler((0, 1))
    data_normalized = scaler.fit_transform(data)
    n_samples = data_normalized.shape[0]
    
    for i in range(n_samples):
        image = data_normalized[i].reshape(image_size, image_size)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{prefix}_cluster_{cluster_label}_image_{i}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

n_samples_per_cluster = 100
low_dim_size = 5
high_dim_size = 80 * 80
image_size = 80

cluster_0_data, cluster_1_data = generate_clusters(n_samples_per_cluster, low_dim_size)
low_dim_data = np.vstack([cluster_0_data, cluster_1_data])
cluster_labels = np.array([0] * n_samples_per_cluster + [1] * n_samples_per_cluster)

linear_mapped_data = linear_mapping(low_dim_data, high_dim_size)
nonlinear_model = NonLinearMapping(input_dim=low_dim_size, output_dim=high_dim_size)
nonlinear_mapped_data = nonlinear_mapping(low_dim_data, nonlinear_model)

#save_images(linear_mapped_data[:n_samples_per_cluster], image_size, prefix="linear", cluster_label=0)
#save_images(linear_mapped_data[n_samples_per_cluster:], image_size, prefix="linear", cluster_label=1)
save_images(nonlinear_mapped_data[:n_samples_per_cluster], image_size, prefix="nonlinear", cluster_label=0)
save_images(nonlinear_mapped_data[n_samples_per_cluster:], image_size, prefix="nonlinear", cluster_label=1)

print(f"Images saved in {output_dir} folder.")
