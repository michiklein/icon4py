uenv start --view=default icon/25.2:v3
source .venv/bin/activate
pip install cupy-cuda12x nvidia-cuda-runtime-cu12==12.6.37
.venv/bin/python3.10 model/common/tests/mklein/mklein_test.py