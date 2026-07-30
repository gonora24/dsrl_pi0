import h5py

path = "/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_NA_noise_action_mapping/noise_action_mapping_task59.h5"

with h5py.File(path, 'r') as f:
    print("n_saved:", f.attrs['n_saved'])
    print("keys:", list(f.keys()))
    for key in f.keys():
        ds = f[key]
        print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")