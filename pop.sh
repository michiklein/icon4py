cd ..
pushd icon4py
    uv sync --no-cache --extra all
    source .venv/bin/activate
popd

pushd gt4py
    uv pip install --no-cache -e .
popd
cd icon4py/
