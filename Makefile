.PHONY: setup data features train app test lint all

setup:
	pip install -e ".[dev]"

data:
	python -m soja_rs.data

features:
	python -m soja_rs.features

train:
	python -m soja_rs.train

app:
	streamlit run src/soja_rs/app.py

test:
	pytest

lint:
	ruff check src tests

all: data features train
