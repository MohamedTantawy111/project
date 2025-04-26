import numpy as np
import pandas as pd
from spectral import open_image, get_rgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def load_hyper_image(hdr_file):
    """Load hyperspectral image with wavelength information"""
    img = open_image(hdr_file)
    data = img.load()
    wavelengths = np.array(img.bands.centers)  # Get wavelength values from HDR
    return data, wavelengths

def save_cluster_spectra(cluster_map, spectral_data, wavelengths, output_dir="cluster_spectra"):
    """Save average spectra for each cluster to CSV files"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Reshape cluster map to match spectral data
    flat_clusters = cluster_map.flatten()
    
    # Get unique clusters
    unique_clusters = np.unique(flat_clusters)
    
    for cluster_id in unique_clusters:
        # Get indices of pixels in this cluster
        mask = (flat_clusters == cluster_id)
        cluster_pixels = spectral_data[mask]
        
        # Calculate average spectrum
        avg_spectrum = np.mean(cluster_pixels, axis=0)
        
        # Create DataFrame with wavelengths
        df = pd.DataFrame({
            'Wavelength(um)': wavelengths,
            'Average_Reflectance': avg_spectrum
        })
        
        # Save to CSV
        filename = os.path.join(output_dir, f'cluster_{cluster_id+1}.csv')
        df.to_csv(filename, index=False)
        print(f'Saved {filename}')

def hyperspectral_clustering(hdr_path, n_clusters=5, n_components=10):
    # Load data and wavelengths
    hyper_img, wavelengths = load_hyper_image(hdr_path)
    original_shape = hyper_img.shape
    
    # Prepare spectral data
    spectral_data = hyper_img.reshape(-1, original_shape[2])
    
    # Preprocessing
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(spectral_data)

    # Dimensionality reduction
    if n_components:
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data_scaled)
    else:
        data_pca = data_scaled

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data_pca)
    cluster_map = clusters.reshape(original_shape[0], original_shape[1])

    return cluster_map, spectral_data, wavelengths

# Example usage
if __name__ == "__main__":
    hdr_path = "tree/tree.hdr"  # Replace with your file path
    
    # Process the data
    cluster_map, spectral_data, wavelengths = hyperspectral_clustering(
        hdr_path,
        n_clusters=5,
        n_components=10
    )
    
    # Save cluster spectra to CSV
    save_cluster_spectra(cluster_map, spectral_data, wavelengths)
    
    # Visualization (from previous code)
    # visualize_results(cluster_map, spectral_data, wavelengths, n_clusters=5)