clean:
	rm -rf build/ dist/ *.egg-info/ target/ .maturin/

develop:
	maturin develop

install:
	maturin develop && \
	pip install .
