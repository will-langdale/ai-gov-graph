#!/usr/bin/env just --justfile

# Default command
default:
    just -l

# Reformat and lint
format:
    uvx ruff@latest format .
    uvx ruff@latest check . --fix

# Run type checking
check *ARGS:
    uv run ty check --output-format concise {{ARGS}}

# Run unit tests
test *ARGS:
    uv run pytest
