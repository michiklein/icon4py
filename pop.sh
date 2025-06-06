uv venv --python $(which python)

cd ..
pushd icon4py
    uv sync --extra all --extra cuda12
    source .venv/bin/activate
popd

pushd gt4py
    uv pip install --no-cache -e .
popd

pushd dace
    uv pip install --no-cache -e .
popd
cd icon4py/
