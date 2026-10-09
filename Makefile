.PHONY: setup submodules data reproduce tables test

setup:
	pip install -r requirements.txt
	pip install -e .

# Run once: adds the upstream repos as git submodules under third_party/.
# After that, pin each to a specific commit and commit the submodule pointer.
submodules:
	git submodule add https://github.com/carrtesy/M2N2 third_party/M2N2
	git submodule add https://github.com/kimanki/CANDI third_party/CANDI

data:
	bash scripts/download_data.sh
	python scripts/preprocess.py

reproduce:
	python scripts/run.py --config configs/experiments/reproduce.yaml

tables:
	python scripts/make_tables.py

test:
	pytest -q
