# Build and run all tests
cargo build
cargo test
cd vesto-core
cargo test --release -- --nocapture
cd ../vesto-python
source venv_vesto/bin/activate
maturin develop
pytest -v --maxfail=1 --disable-warnings