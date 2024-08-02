import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def create_distance_features(vectors, centroid, query_vector, alpha=0.5, method='concatenate', degree=2):
    distances_to_centroid = np.linalg.norm(vectors - centroid, axis=1) * (1. - alpha)
    distances_to_query = np.linalg.norm(vectors - query_vector, axis=1) * alpha

    # Reshape distances
    distances_to_centroid = distances_to_centroid.reshape(-1, 1)
    distances_to_query = distances_to_query.reshape(-1, 1)

    if method == 'concatenate':
        features = np.column_stack((distances_to_centroid, distances_to_query))
    elif method == 'weighted_sum':
        features = (alpha * distances_to_query + (1 - alpha) * distances_to_centroid).reshape(-1, 1)
    elif method == 'interaction':
        product_distances = (distances_to_centroid * distances_to_query).reshape(-1, 1)
        ratio_distances = (distances_to_centroid / (distances_to_query + 1e-8)).reshape(-1, 1)
        features = np.column_stack((distances_to_centroid, distances_to_query, product_distances, ratio_distances))
    elif method == 'polynomial':
        combined_distances = np.hstack((distances_to_centroid, distances_to_query))
        poly = PolynomialFeatures(degree)
        features = poly.fit_transform(combined_distances)
    else:
        raise ValueError("Unsupported method. Choose from 'concatenate', 'weighted_sum', 'interaction', or 'polynomial'.")

    print_features_shape(vectors, distances_to_centroid, distances_to_query, features)

    return features

def print_features_shape(vectors, distances_to_centroid, distances_to_query, features):
    print("Shape of vectors:", vectors.shape)
    print("Shape of distances_to_centroid:", distances_to_centroid.shape)
    print("Shape of distances_to_query:", distances_to_query.shape)
    print("Shape of combined features:", features.shape)


def transform_and_plot(vectors, ids, labels, log_likelihood, threshold, centroid, query_vector, alpha=0.5,
                       method='concatenate', degree=2, plot_file=None, title='PCA of Distance Features'):
    features = create_distance_features(vectors, centroid, query_vector, alpha, method, degree)

    # Standardize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Apply PCA for dimensionality reduction if there are at least 2 features
    if features_normalized.shape[1] > 1:
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_normalized)
    else:
        features_pca = features_normalized  # Skip PCA if there is only one feature

    # Calculate distances for the centroid and query vector using the same method
    distances_to_centroid = np.linalg.norm(vectors - centroid, axis=1)
    distances_to_query = np.linalg.norm(vectors - query_vector, axis=1)

    if method == 'concatenate':
        centroid_distances = np.array([np.mean(distances_to_centroid), np.mean(distances_to_query)])
        query_distances = np.array([np.mean(distances_to_query), 0])
    elif method == 'weighted_sum':
        centroid_distances = np.array(
            [alpha * np.mean(distances_to_query) + (1 - alpha) * np.mean(distances_to_centroid)])
        query_distances = np.array([alpha * np.mean(distances_to_query)])
    elif method == 'interaction':
        product_distances = distances_to_centroid * distances_to_query
        ratio_distances = distances_to_centroid / (distances_to_query + 1e-8)
        centroid_distances = np.array(
            [np.mean(distances_to_centroid), np.mean(distances_to_query), np.mean(product_distances),
             np.mean(ratio_distances)])
        query_distances = np.array(
            [np.mean(distances_to_query), 0, np.mean(product_distances), np.mean(ratio_distances)])
    elif method == 'polynomial':
        combined_distances = np.hstack((distances_to_centroid.reshape(-1, 1), distances_to_query.reshape(-1, 1)))
        poly = PolynomialFeatures(degree)
        combined_distances_poly = poly.fit_transform(combined_distances)
        centroid_distances = np.mean(combined_distances_poly, axis=0)
        query_distances = np.zeros_like(centroid_distances)
        query_distances[0] = 1  # The first feature of polynomial features is always 1 (bias term)
    else:
        raise ValueError(
            "Unsupported method. Choose from 'concatenate', 'weighted_sum', 'interaction', or 'polynomial'.")

    # Standardize these distances using the same scaler
    centroid_distances_standardized = scaler.transform([centroid_distances])
    query_distances_standardized = scaler.transform([query_distances])

    # Apply PCA transformation to the standardized distances if applicable
    if features_normalized.shape[1] > 1:
        centroid_pca = pca.transform(centroid_distances_standardized)
        query_pca = pca.transform(query_distances_standardized)
    else:
        centroid_pca = centroid_distances_standardized
        query_pca = query_distances_standardized

    # Plotting
    fig = plt.figure(figsize=(10, 7))

    # Plot clusters
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = features_pca[labels == label]
        if features_pca.shape[1] > 1:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
        else:
            plt.scatter(cluster_points, np.zeros_like(cluster_points), label=f'Cluster {label}')

    # Plot outliers with a bigger red star
    outlier_points = features_pca[log_likelihood < threshold]
    if features_pca.shape[1] > 1:
        plt.scatter(outlier_points[:, 0], outlier_points[:, 1], c='red', marker='*', s=200, label='Outliers')
    else:
        plt.scatter(outlier_points, np.zeros_like(outlier_points), c='red', marker='*', s=200, label='Outliers')

    # Annotate points with IDs
    for i, txt in enumerate(ids):
        if features_pca.shape[1] > 1:
            plt.annotate(txt, (features_pca[i, 0], features_pca[i, 1]))
        else:
            plt.annotate(txt, (features_pca[i], 0))

    if features_pca.shape[1] > 1:
        plt.scatter(centroid_pca[0, 0], centroid_pca[0, 1], c='red', marker='X', s=100, label='Centroid')
        plt.scatter(query_pca[0, 0], query_pca[0, 1], c='blue', marker='^', s=100, label='Query Vector')
    else:
        plt.scatter(centroid_pca, 0, c='red', marker='X', s=100, label='Centroid')
        plt.scatter(query_pca, 0, c='blue', marker='^', s=100, label='Query Vector')

    plt.xlabel('PCA Component 1' if features_pca.shape[1] > 1 else 'Feature 1')
    if features_pca.shape[1] > 1:
        plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if plot_file:
        pdf = backend_pdf.PdfPages(plot_file)
        pdf.savefig(fig)
        pdf.close()
    else:
        plt.show()