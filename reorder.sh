#!/bin/bash

# Simple batch reorder script that definitely works
# Usage: ./simple_batch_reorder.sh /path/to/folder

if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_path>"
    echo "Example: $0 /path/to/nc/files"
    exit 1
fi

FOLDER="$1"

if [ ! -d "$FOLDER" ]; then
    echo "Error: Directory '$FOLDER' does not exist"
    exit 1
fi

echo "Looking for .nc files in: $FOLDER"

# Find .nc files (excluding already reordered ones)
NC_FILES=($(find "$FOLDER" -name "*.nc" -not -name "*_reordered.nc" | sort))

if [ ${#NC_FILES[@]} -eq 0 ]; then
    echo "No .nc files found in $FOLDER"
    exit 1
fi

echo "Found ${#NC_FILES[@]} files to process:"
for file in "${NC_FILES[@]}"; do
    echo "  - $(basename "$file")"
done

echo ""
echo "Starting submissions..."

# Create reorder directory
mkdir -p reorder

TIME_LIMIT=15
MIN_TIME=1

for i in "${!NC_FILES[@]}"; do
    file="${NC_FILES[$i]}"
    basename_file=$(basename "$file" .nc)
    job_name="reorder_${basename_file}"
    job_name=$(echo "$job_name" | sed 's/[^a-zA-Z0-9_-]/_/g' | cut -c1-50)
    
    echo "Submitting: $(basename "$file") (${TIME_LIMIT}h)"
    
    # Submit job directly
    JOB_ID=$(sbatch --parsable << EOF
#!/bin/bash -l
#SBATCH --time=${TIME_LIMIT}:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -A d75
#SBATCH --exclusive
#SBATCH --job-name=${job_name}
#SBATCH --output=reorder/${job_name}_%j.out
#SBATCH --error=reorder/${job_name}_%j.err
#SBATCH --uenv=icon/25.2:v3 --view=default

export GT4PY_BUILD_CACHE_DIR="\$(pwd)/GT4PYcache"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export CUDAARCHS=90

mkdir -p nsys
mkdir -p reorder
source .venv/bin/activate

echo "Starting grid reordering for: ${file}"
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"

python model/common/tests/mklein/grid_reorder.py "${file}"

echo "Job completed at: \$(date)"
EOF
)
    
    if [ $? -eq 0 ]; then
        echo "  Job ID: $JOB_ID"
    else
        echo "  FAILED to submit"
    fi
    
    # Decrement time limit
    if [ $TIME_LIMIT -gt $MIN_TIME ]; then
        TIME_LIMIT=$((TIME_LIMIT - 1))
    fi
    
    sleep 1
done

echo ""
echo "All jobs submitted! Check with: squeue -u \$USER"