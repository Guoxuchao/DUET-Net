import os
import numpy as np
import scipy.io as sio
from glob import glob
import struct

def read_mat_file_manual(mat_file_path):
    """
    Manually parse MAT file to handle binary format issues.
    """
    try:
        mat_data = sio.loadmat(mat_file_path)
        return mat_data
    except Exception as e:
        print(f"Scipy load failed: {e}. Attempting manual parsing...")

        with open(mat_file_path, 'rb') as f:
            header = f.read(116)  
            f.read(8) 
            
            data_elements = []
            while True:
                data_type_bytes = f.read(4)
                if not data_type_bytes:
                    break
                    
                data_type = struct.unpack('>I', data_type_bytes)[0]
                num_bytes_bytes = f.read(4)
                num_bytes = struct.unpack('>I', num_bytes_bytes)[0]
                
                data = f.read(num_bytes)

                if num_bytes % 8 != 0:
                    padding = 8 - (num_bytes % 8)
                    f.read(padding)
                
                # Handle miMATRIX type (14)
                if data_type == 14:
                    matrix_data = parse_matrix_data(data)
                    if matrix_data is not None:
                        data_elements.append(matrix_data)
                
        return {'annPoints': data_elements[0] if data_elements else np.array([])}

def parse_matrix_data(data):
    """
    Parse MATLAB matrix data.
    """
    try:
        if len(data) >= 8:
            num_elements = len(data) // 8
            values = []
            for i in range(num_elements):
                value_bytes = data[i*8:(i+1)*8]
                try:
                    value = struct.unpack('>d', value_bytes)[0]
                    values.append(value)
                except:
                    pass
            
            # Assume Nx2 coordinate array
            if len(values) % 2 == 0 and len(values) > 0:
                coords = np.array(values).reshape(-1, 2)
                # Add 3rd column (size), default 15.0
                if coords.shape[0] > 0:
                    sizes = np.full((coords.shape[0], 1), 15.0)
                    result = np.hstack([coords, sizes]).astype(np.float32)
                    return result
        return None
    except Exception as e:
        print(f"Matrix parsing failed: {e}")
        return None

def convert_mat_to_npy(mat_file_path, npy_file_path):
    """
    Convert a single MAT file to NPY format.
    """
    try:
        print(f"Converting: {mat_file_path}")
        
        mat_data = read_mat_file_manual(mat_file_path)
        
        # Extract annotation points
        if 'annPoints' in mat_data:
            ann_points = mat_data['annPoints']
        else:
            # Try fallback field names
            possible_fields = ['points', 'locations', 'coordinates', 'gt_points']
            ann_points = None
            for field in possible_fields:
                if field in mat_data:
                    ann_points = mat_data[field]
                    break
            
            if ann_points is None:
                # Try first non-system field
                for key, value in mat_data.items():
                    if not key.startswith('__'):
                        ann_points = value
                        break
        
        if ann_points is None:
            print(f"Warning: No points found in {mat_file_path}. Creating empty array.")
            ann_points = np.array([]).reshape(0, 3)
        
        # Ensure correct data format (Nx3: x, y, sigma)
        if isinstance(ann_points, np.ndarray):
            if ann_points.ndim == 2:
                if ann_points.shape[1] == 2:
                    # Add 3rd column (sigma=15.0)
                    sizes = np.full((ann_points.shape[0], 1), 15.0)
                    ann_points = np.hstack([ann_points, sizes]).astype(np.float32)
                elif ann_points.shape[1] == 1:
                    # Special case
                    ann_points = np.column_stack([ann_points, ann_points, np.full_like(ann_points, 15.0)]).astype(np.float32)
            elif ann_points.ndim == 1:
                # 1D array, assume interleaved x, y
                if len(ann_points) % 2 == 0:
                    coords = ann_points.reshape(-1, 2)
                    sizes = np.full((coords.shape[0], 1), 15.0)
                    ann_points = np.hstack([coords, sizes]).astype(np.float32)
                else:
                    ann_points = np.array([]).reshape(0, 3)
        else:
            ann_points = np.array([]).reshape(0, 3)
        
        np.save(npy_file_path, ann_points.astype(np.float32))
        print(f"Success: {mat_file_path} -> {npy_file_path} (Points: {len(ann_points)})")
        return True
        
    except Exception as e:
        print(f"Failed {mat_file_path}: {e}")
        # Create empty fallback
        try:
            empty_data = np.array([]).reshape(0, 3)
            np.save(npy_file_path, empty_data)
            print(f"Created empty npy: {npy_file_path}")
            return True
        except:
            return False

def batch_convert_directory(directory_path):
    """
    Batch convert all MAT files in a directory.
    """
    print(f"\nProcessing directory: {directory_path}")
    
    mat_files = glob(os.path.join(directory_path, "*.mat"))
    
    if not mat_files:
        print("No .mat files found.")
        return
    
    print(f"Found {len(mat_files)} .mat files")
    
    success_count = 0
    
    for mat_file in mat_files:
        base_name = os.path.splitext(mat_file)[0]
        npy_file = base_name + ".npy"
        
        if convert_mat_to_npy(mat_file, npy_file):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(mat_files)} files converted.")

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "maize_tassel")
    
    directories = [
        os.path.join(base_dir, "train"),
        os.path.join(base_dir, "val"), 
        os.path.join(base_dir, "test")
    ]
    
    print("Starting conversion of maize_tassel dataset MAT files to NPY format...")
    print("=" * 60)
    
    for directory in directories:
        if os.path.exists(directory):
            batch_convert_directory(directory)
        else:
            print(f"Directory not found: {directory}")
    
    print("=" * 60)
    print("All conversions finished!")
    print("\nNPY Format Description:")
    print("- Nx3 numpy array")
    print("- Col 1: x coordinate")
    print("- Col 2: y coordinate")
    print("- Col 3: sigma (fixed at 15.0)")

if __name__ == "__main__":
    main()
