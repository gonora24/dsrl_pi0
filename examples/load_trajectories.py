import h5py

with h5py.File("logs/collect_pi05_libero_90_task_2026_07_15_16_58_28_0000--s-0/trajectories.hdf5", "r") as f:
    success_count = 0
    total_count = 0
    for key in f["data"].keys():
        demo = f["data"][key]
        success = demo.attrs["is_success"]
        if success:
            success_count += 1
        total_count += 1

print(f"Success count: {success_count}")
print(f"Success rate: {success_count / total_count}")
print(f"Total count: {total_count}")
