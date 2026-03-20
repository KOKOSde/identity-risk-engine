PYTHON ?= python3.11

.PHONY: test lint demo

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

demo:
	$(PYTHON) -m nbconvert --to notebook --execute notebooks/demo.ipynb --output demo.executed.ipynb --output-dir notebooks
