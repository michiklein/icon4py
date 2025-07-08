#!/bin/bash -l
#SBATCH --time=4:00:00
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

rm -rf GT4PYcacheon
export GT4PY_BUILD_CACHE_DIR="$(pwd)/GT4PYcacheon"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export CUDAARCHS=90
mkdir -p ncu
mkdir -p nsys
source .venv/bin/activate
export CUDAFLAGS="--generate-line-info"
export GT4PY_ENABLE_COLLAPSE_TABLES=1

# srun ncu -o ncu/v2c2e_on --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2c2e.py
# srun ncu -o ncu/v2c2v_on --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2c2v.py
# srun ncu -o ncu/v2e2c_on --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2e2c.py
# srun ncu -o ncu/v2e2v_on --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2e2v.py

srun nsys profile --stats=true --force-overwrite=true -o nsys/v2c2e_on .venv/bin/python3.10 model/common/tests/mklein/v2c2e.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/v2c2v_on .venv/bin/python3.10 model/common/tests/mklein/v2c2v.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/v2e2c_on .venv/bin/python3.10 model/common/tests/mklein/v2e2c.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/v2e2v_on .venv/bin/python3.10 model/common/tests/mklein/v2e2v.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/c2e2c_on .venv/bin/python3.10 model/common/tests/mklein/c2e2c.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/c2e2v_on .venv/bin/python3.10 model/common/tests/mklein/c2e2v.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/c2v2c_on .venv/bin/python3.10 model/common/tests/mklein/c2v2c.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/c2v2e_on .venv/bin/python3.10 model/common/tests/mklein/c2v2e.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/e2c2v_on .venv/bin/python3.10 model/common/tests/mklein/e2c2v.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/e2v2c_on .venv/bin/python3.10 model/common/tests/mklein/e2v2c.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/e2v2e_on .venv/bin/python3.10 model/common/tests/mklein/e2v2e.py
