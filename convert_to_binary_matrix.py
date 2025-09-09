import numpy as np
import matplotlib.pyplot as plt
import argparse


# Load files
def load_data(obs_path, exp_path):
    raw_obs = np.loadtxt(obs_path) # 2v2 matrix, value is num count
    raw_exp = np.loadtxt(exp_path).ravel() #1d vector
    print("RAW observed shape:", raw_obs.shape, ". RAW expected shape:", raw_exp.shape)
    return raw_obs, raw_exp

# Convert to binary adjacency matrix
def convert_to_binary_matrix(raw_obs, raw_exp, bin_size=1_000_000, p=95):
    # Bin indices and values
    i = (raw_obs[:,0] // bin_size).astype(int) #bin indices
    j = (raw_obs[:,1] // bin_size).astype(int) #bin indices
    v = raw_obs[:,2] #values
    # Create sparse matrix
    n = int(max(i.max(), j.max())) + 1   # number of bins (expect ~250 for chr1 at 1Mb)
    obs = np.zeros((n, n), dtype=float)
    np.add.at(obs, (i, j), v)
    np.add.at(obs, (j, i), v)            # symmetrize (in case file is upper-tri only)
    # optional: ensure diagonal present
    obs[np.arange(n), np.arange(n)] += 0

    exp_full = np.r_[np.nan, raw_exp]                     # now length should be n (=250)
    # Build expected matrix by distance and compute O/E 
    idx_i = np.arange(n)[:, None] # column vector
    idx_j = np.arange(n)[None, :] # row vector
    dist = np.abs(idx_i - idx_j)  # values in 0..n-1
    exp_mat = exp_full[dist]  
    OE = obs / exp_mat
    OE[~np.isfinite(OE)] = np.nan           # clean inf/NaN
    np.fill_diagonal(OE, np.nan)            # ignore self-self
    # Threshold O/E to get binary adjacency matrix
    tau = np.nanpercentile(OE, p) # threshold value
    A = (OE >= tau).astype(int)
    A[np.isnan(OE)] = 0
    np.fill_diagonal(A, 0)
    print("tau ({}th pct):".format(p), tau, "edges:", A.sum()//2)
    return A

def main():
    #Sample run command: 
    # convert_to_binary_matrix.py --obs_path MAPQGE30/chr1_1mb.RAWobserved --exp_path MAPQGE30/chr1_1mb.RAWexpected --bin_size 1000000 --percentile 95 --output_path chr1_1mb_A.npy  
    parser = argparse.ArgumentParser(description='Convert Hi-C data to binary adjacency matrix')
    parser.add_argument("--obs_path", help = "input path to RAWobserved file" )
    parser.add_argument("--exp_path", help = "input path to RAWexpected file" )
    parser.add_argument("--bin_size", type=int, default=1_000_000, help = "bin size in base pairs (default: 1,000,000)" )
    parser.add_argument("--percentile", type=int, default=95, help = "percentile threshold for adjacency (default: 95)" )
    parser.add_argument("--output_path", help = "output path to save binary adjacency matrix (npy format)" )
    args = parser.parse_args()

    raw_obs, raw_exp = load_data(args.obs_path, args.exp_path)
    A = convert_to_binary_matrix(raw_obs, raw_exp, bin_size=args.bin_size, p=args.percentile)
    print("Binary adjacency matrix shape:", A.shape)

    if args.output_path:
        np.savetxt(args.output_path, A, fmt="%d")
        print("Binary adjacency matrix saved to", args.output_path)
    else:
        print("No output path provided. Binary adjacency matrix not saved.")

if __name__ == "__main__":
    main()
