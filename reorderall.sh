#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

FOLDER="$1"
if [ ! -d "$FOLDER" ]; then
    echo "Error: Directory '$FOLDER' does not exist"
    exit 1
fi

echo "Submitting batch reordering job for folder: $FOLDER"

JOB_ID=$(sbatch --parsable << EOF
#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -A d75
#SBATCH --exclusive
#SBATCH --job-name=reorder_batch
#SBATCH --output=reorder_batch_%j.out
#SBATCH --uenv=icon/25.2:v3 --view=default

export GT4PY_BUILD_CACHE_DIR="\$(pwd)/GT4PYcache"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export CUDAARCHS=90

source .venv/bin/activate

python << 'PYTHON_EOF'
import os
import time
import glob
import sys
sys.path.append('model/common/tests/mklein')
from grid_reorder import reorder_grid_file

folder = '$FOLDER'
nc_files = glob.glob(os.path.join(folder, "*.nc"))
nc_files = [f for f in nc_files if not f.endswith('_reordered.nc')]
nc_files.sort(key=lambda f: os.path.getsize(f), reverse=True)

total_start = time.time()
print(f"Processing {len(nc_files)} files...")

for i, nc_file in enumerate(nc_files, 1):
    basename = os.path.basename(nc_file)
    reordered_file = nc_file.replace('.nc', '_reordered.nc')
    
    if os.path.exists(reordered_file):
        print(f"[{i}/{len(nc_files)}] Skipping {basename} (already exists)")
        continue
    
    print(f"[{i}/{len(nc_files)}] Processing {basename}")
    
    file_start = time.time()
    try:
        reorder_grid_file(nc_file)
        elapsed = time.time() - file_start
        print(f"  Completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - file_start
        print(f"  FAILED after {elapsed:.1f}s: {e}")

total_elapsed = time.time() - total_start
print(f"Total time: {total_elapsed:.1f}s")
PYTHON_EOF
EOF
)

if [ $? -eq 0 ]; then
    echo "Job submitted successfully with ID: $JOB_ID"
    echo "Monitor with: squeue -j $JOB_ID"
else
    echo "Failed to submit job"
fi