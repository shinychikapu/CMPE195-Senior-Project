import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

# Path to your Hi-C raw observed data file
filepath = '/Users/kylewang/Downloads/GM12878_primary 2/1mb_resolution_intrachromosomal/chr1/MAPQG0/chr1_1mb.RAWobserved'

bin_size = 1000000  # 1 Mb resolution

rows = []
cols = []
data = []

# Read the file and bin genomic coordinates
with open(filepath, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        i, j, val = parts
        i = int(i) // bin_size
        j = int(j) // bin_size
        val = float(val)
        rows.append(i)
        cols.append(j)
        data.append(val)

# Build sparse matrix from binned data
size = max(max(rows), max(cols)) + 1
sparse_mat = coo_matrix((data, (rows, cols)), shape=(size, size))

# Make the matrix symmetric by adding its transpose
sparse_mat_symmetric = sparse_mat + sparse_mat.T

# Fix diagonal (if double counted)
sparse_mat_symmetric.setdiag(sparse_mat.diagonal())

# Convert sparse matrix to dense array
matrix = sparse_mat_symmetric.toarray()

# Apply log transform for better color scaling
matrix_log = np.log1p(matrix)  # log(1 + x)

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(matrix_log, cmap='OrRd', origin='lower')

# Add colorbar
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label='log(1 + contact count)')

# Labels and title
ax.set_xlabel('Genomic bin index')
ax.set_ylabel('Genomic bin index')
ax.set_title('Hi-C Contact Heatmap (Chr1, 1Mb bins)')

# Show fewer ticks for readability
num_bins = matrix.shape[0]
tick_spacing = max(num_bins // 10, 1)
ax.set_xticks(range(0, num_bins, tick_spacing))
ax.set_yticks(range(0, num_bins, tick_spacing))

plt.tight_layout()
plt.show()
