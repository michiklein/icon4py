#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -A d75
#SBATCH --exclusive
#SBATCH --uenv=icon/25.2:v3 --view=default

start_time=$(date +%s)

rm -rf GT4PYcacheon_32
export GT4PY_BUILD_CACHE_DIR="$(pwd)/GT4PYcacheon_32"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export CUDAARCHS=90
mkdir -p ncu
mkdir -p nsys32
source .venv/bin/activate
export CUDAFLAGS="--generate-line-info"
export GT4PY_ENABLE_COLLAPSE_TABLES=1
export GT4PY_COLLAPSE_TABLES_BLOCK_32=1

# srun ncu -o ncu/v2c2e_on --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2c2e.py --block-sort
# srun ncu -o ncu/v2c2v_on --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2c2v.py --block-sort
# srun ncu -o ncu/v2e2c_on --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2e2c.py --block-sort
# srun ncu -o ncu/v2e2v_on --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2e2v.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/c2e2c2e2c_on_32 .venv/bin/python3.10 model/common/tests/mklein/c2e2c2e2c.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/v2e2c2v_on_32 .venv/bin/python3.10 model/common/tests/mklein/v2e2c2v.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/v2c2e_on_32 .venv/bin/python3.10 model/common/tests/mklein/v2c2e.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/v2c2v_on_32 .venv/bin/python3.10 model/common/tests/mklein/v2c2v.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/v2e2c_on_32 .venv/bin/python3.10 model/common/tests/mklein/v2e2c.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/v2e2v_on_32 .venv/bin/python3.10 model/common/tests/mklein/v2e2v.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/c2e2c_on_32 .venv/bin/python3.10 model/common/tests/mklein/c2e2c.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/c2e2v_on_32 .venv/bin/python3.10 model/common/tests/mklein/c2e2v.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/c2v2c_on_32 .venv/bin/python3.10 model/common/tests/mklein/c2v2c.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/c2v2e_on_32 .venv/bin/python3.10 model/common/tests/mklein/c2v2e.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/e2c2v_on_32 .venv/bin/python3.10 model/common/tests/mklein/e2c2v.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/e2v2c_on_32 .venv/bin/python3.10 model/common/tests/mklein/e2v2c.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/e2v2e_on_32 .venv/bin/python3.10 model/common/tests/mklein/e2v2e.py --block-sort
srun nsys profile --stats=true --force-overwrite=true -o nsys32/e2v2c_on_32 .venv/bin/python3.10 model/common/tests/mklein/e2v2c.py --block-sort


end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s"