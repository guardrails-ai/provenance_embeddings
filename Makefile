dev:
	pip install -e ".[dev]"
	python3 validator/post-install.py

lint:
	ruff check .

tests:
	pytest ./test

type:
	pyright validator

qa:
	make lint
	make type
	make tests