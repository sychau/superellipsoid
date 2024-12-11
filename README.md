# Dependencies
- trimesh
- numpy
- pytorch
- SciPy
- sympy
- Cython
- pyquaternion

Compile fast sampler by
```bash
python build.py build_ext --inplace
```

Training and visualization
```bash
python main.py
```
A pyd file with specific system version should be generated in fast_sampler, e.g. "_sampler.cp311-win_amd64.pyd"

If vscode show warning, add your global package site path to setting Python › Analysis: Extra Paths

voxel_and_sdf.npz contains five items: voxels, sdf_points, sdf_values, centroid, scale

voxels is a 64x64x64 binary occupancy grid
sdf_points: 100000 points
centroid: 3x1 array
scale: a number