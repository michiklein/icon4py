GT4PY_BUILD_CACHE_DIR=$(pwd)/GT4PYcache1
GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=$(pwd)/GT4PYcache1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export CUDAARCHS=90
export GT4PY_ENABLE_COLLAPSE_TABLES=1

# .venv/bin/python3.10 model/common/tests/mklein/mklein_test.py