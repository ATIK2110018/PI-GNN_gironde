"""
Compares the FVM mesh definition file (FlowFM_net.nc) against the 
model output file to verify node/cell consistency.
"""
import netCDF4 as nc
import numpy as np

MESH_FILE = r"c:\Users\atikr\Desktop\hydrodynamic\hydrodynamic_calibration - Copy\calibration.dsproj_data\FlowFM\data\input\FlowFM_net.nc"

print("=" * 60)
print("MESH FILE: FlowFM_net.nc")
print("=" * 60)
with nc.Dataset(MESH_FILE, 'r') as ds:
    print("\nVariables in mesh file:")
    for varname, var in ds.variables.items():
        print(f"  {varname:40s} shape={var.shape}  dims={var.dimensions}")
    
    # Common Delft3D FM variable names for nodes and faces
    node_vars = ['NetNode_x', 'NetNode_y', 'NetNode_z', 
                 'mesh2d_node_x', 'mesh2d_node_y',
                 'Mesh2D_node_x', 'Mesh2D_node_y']
    face_vars = ['NetElemNode', 'mesh2d_face_x', 'mesh2d_face_y',
                 'Mesh2D_face_x', 'Mesh2D_face_y',
                 'NetElem_x', 'NetElem_y']
    
    print("\n--- Node (vertex) coordinates ---")
    for v in node_vars:
        if v in ds.variables:
            data = ds.variables[v][:]
            print(f"  {v}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")
    
    print("\n--- Face (cell) coordinates ---")
    for v in face_vars:
        if v in ds.variables:
            data = ds.variables[v][:]
            print(f"  {v}: shape={data.shape}")

    print(f"\nDimensions:")
    for dname, dim in ds.dimensions.items():
        print(f"  {dname}: {len(dim)}")

print("\n" + "=" * 60)
print("Looking for the OUTPUT .nc file...")
print("=" * 60)

import os
# Search for map output file locally
search_dirs = [
    r"c:\Users\atikr\Desktop\hydrodynamic\hydrodynamic_calibration - Copy",
    r"c:\Users\atikr\Desktop\hydrodynamic"
]

for d in search_dirs:
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith('_map.nc') or f.endswith('_his.nc'):
                fpath = os.path.join(root, f)
                size_mb = os.path.getsize(fpath) / (1024*1024)
                print(f"  Found: {fpath}  ({size_mb:.1f} MB)")
                if size_mb > 1:  # Only inspect larger files (actual output)
                    with nc.Dataset(fpath, 'r') as ds2:
                        print(f"    Variables: {list(ds2.variables.keys())[:10]}")
                        print(f"    Dimensions: {dict((k, len(v)) for k, v in ds2.dimensions.items())}")
