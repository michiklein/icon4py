#!/bin/bash -l
#SBATCH --time=12:00:00
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

#rm -rf GT4PYcacheoff
export GT4PY_BUILD_CACHE_DIR="$(pwd)/GT4PYcacheoff"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export CUDAARCHS=90
mkdir -p ncu
mkdir -p nsys
source .venv/bin/activate
export CUDAFLAGS="--generate-line-info"
export GT4PY_ENABLE_COLLAPSE_TABLES=0
export GT4PY_COLLAPSE_TABLES_BLOCK_32=0

#srun ncu -o ncu/v2c2e_off --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2c2e.py
#srun ncu -o ncu128/v2c2v_off --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2c2v.py
#srun ncu -o ncu/v2e2c2v_off --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/v2e2c2v.py
#srun ncu -o ncu/c2e2c2e2c_off --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/c2e2c2e2c.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/c2e2c2e2c_off .venv/bin/python3.10 model/common/tests/mklein/c2e2c2e2c.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/v2e2c2v_off .venv/bin/python3.10 model/common/tests/mklein/v2e2c2v.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/v2c2e_off .venv/bin/python3.10 model/common/tests/mklein/v2c2e.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/v2c2v_off .venv/bin/python3.10 model/common/tests/mklein/v2c2v.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/v2e2c_off .venv/bin/python3.10 model/common/tests/mklein/v2e2c.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/v2e2v_off .venv/bin/python3.10 model/common/tests/mklein/v2e2v.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/c2e2c_off .venv/bin/python3.10 model/common/tests/mklein/c2e2c.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/c2e2v_off .venv/bin/python3.10 model/common/tests/mklein/c2e2v.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/c2v2c_off .venv/bin/python3.10 model/common/tests/mklein/c2v2c.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/c2v2e_off .venv/bin/python3.10 model/common/tests/mklein/c2v2e.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/e2c2v_off .venv/bin/python3.10 model/common/tests/mklein/e2c2v.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/e2v2c_off .venv/bin/python3.10 model/common/tests/mklein/e2v2c.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/e2v2e_off .venv/bin/python3.10 model/common/tests/mklein/e2v2e.py
# srun nsys profile --stats=true --force-overwrite=true -o nsys/e2v2c_off .venv/bin/python3.10 model/common/tests/mklein/e2v2c.py


#srun ncu -o ncu/calculate_nabla2_for_theta_off --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/calculate_nabla2_for_theta.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/calculate_nabla2_for_theta_off .venv/bin/python3.10 model/common/tests/mklein/calculate_nabla2_for_theta.py
#srun ncu -o ncu/edge_diagnostics_off --set full --import-source=on -k kernel .venv/bin/python3.10 model/common/tests/mklein/compute_edge_diagnostics_for_velocity_advection.py
srun nsys profile --stats=true --force-overwrite=true -o nsys/edge_diagnostics_off .venv/bin/python3.10 model/common/tests/mklein/compute_edge_diagnostics_for_velocity_advection.py

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s"
