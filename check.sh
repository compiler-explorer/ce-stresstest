#!/bin/sh

poetry run black src/
poetry run ruff check src/
poetry run mypy src/
