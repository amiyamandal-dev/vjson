.PHONY: all build build-release dev install test test-rust test-python bench clean lint format check docs help

# Default target
all: build

# Development build (debug mode, faster compilation)
build:
	maturin develop

# Release build (optimized)
build-release:
	maturin develop --release

# Alias for build
dev: build

# Install in development mode
install: build

# Run all tests
test: test-rust test-python

# Run Rust tests
test-rust:
	cargo test --all-features

# Run Python tests
test-python: build
	python -m pytest tests/ -v

# Run benchmarks
bench:
	cargo bench

# Clean build artifacts
clean:
	cargo clean
	rm -rf dist/ target/ *.egg-info .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Lint code
lint: lint-rust lint-python

lint-rust:
	cargo clippy --all-features -- -D warnings
	cargo fmt -- --check

lint-python:
	ruff check python/ tests/
	mypy python/ --ignore-missing-imports

# Format code
format: format-rust format-python

format-rust:
	cargo fmt

format-python:
	ruff format python/ tests/
	ruff check --fix python/ tests/

# Check everything (lint + test)
check: lint test

# Build documentation
docs:
	cargo doc --no-deps --open

# Build wheel for distribution
wheel:
	maturin build --release

# Build wheels for all platforms (requires CI)
wheel-all:
	maturin build --release --strip

# Version management
version:
	@grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'

bump-patch:
	@echo "Bumping patch version..."
	@CURRENT=$$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'); \
	MAJOR=$$(echo $$CURRENT | cut -d. -f1); \
	MINOR=$$(echo $$CURRENT | cut -d. -f2); \
	PATCH=$$(echo $$CURRENT | cut -d. -f3); \
	NEW_PATCH=$$((PATCH + 1)); \
	NEW_VERSION="$$MAJOR.$$MINOR.$$NEW_PATCH"; \
	sed -i.bak "s/^version = \"$$CURRENT\"/version = \"$$NEW_VERSION\"/" Cargo.toml && rm Cargo.toml.bak; \
	echo "Version bumped: $$CURRENT -> $$NEW_VERSION"

bump-minor:
	@echo "Bumping minor version..."
	@CURRENT=$$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'); \
	MAJOR=$$(echo $$CURRENT | cut -d. -f1); \
	MINOR=$$(echo $$CURRENT | cut -d. -f2); \
	NEW_MINOR=$$((MINOR + 1)); \
	NEW_VERSION="$$MAJOR.$$NEW_MINOR.0"; \
	sed -i.bak "s/^version = \"$$CURRENT\"/version = \"$$NEW_VERSION\"/" Cargo.toml && rm Cargo.toml.bak; \
	echo "Version bumped: $$CURRENT -> $$NEW_VERSION"

bump-major:
	@echo "Bumping major version..."
	@CURRENT=$$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'); \
	MAJOR=$$(echo $$CURRENT | cut -d. -f1); \
	NEW_MAJOR=$$((MAJOR + 1)); \
	NEW_VERSION="$$NEW_MAJOR.0.0"; \
	sed -i.bak "s/^version = \"$$CURRENT\"/version = \"$$NEW_VERSION\"/" Cargo.toml && rm Cargo.toml.bak; \
	echo "Version bumped: $$CURRENT -> $$NEW_VERSION"

# Release workflow: bump version, commit, push (triggers CI/CD)
release-patch: bump-patch
	@VERSION=$$(make version); \
	git add Cargo.toml; \
	git commit -m "chore: bump version to $$VERSION"; \
	git push

release-minor: bump-minor
	@VERSION=$$(make version); \
	git add Cargo.toml; \
	git commit -m "chore: bump version to $$VERSION"; \
	git push

release-major: bump-major
	@VERSION=$$(make version); \
	git add Cargo.toml; \
	git commit -m "chore: bump version to $$VERSION"; \
	git push

# Show help
help:
	@echo "vjson - High-performance vector database"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  build           Build development version (debug)"
	@echo "  build-release   Build optimized release version"
	@echo "  install         Install in development mode"
	@echo "  test            Run all tests (Rust + Python)"
	@echo "  test-rust       Run Rust tests only"
	@echo "  test-python     Run Python tests only"
	@echo "  bench           Run benchmarks"
	@echo "  lint            Lint all code"
	@echo "  format          Format all code"
	@echo "  check           Run lint + tests"
	@echo "  docs            Build and open documentation"
	@echo "  wheel           Build distribution wheel"
	@echo "  clean           Remove build artifacts"
	@echo ""
	@echo "Versioning:"
	@echo "  version         Show current version"
	@echo "  bump-patch      Bump patch version (0.1.0 -> 0.1.1)"
	@echo "  bump-minor      Bump minor version (0.1.0 -> 0.2.0)"
	@echo "  bump-major      Bump major version (0.1.0 -> 1.0.0)"
	@echo "  release-patch   Bump patch, commit, and push (triggers release)"
	@echo "  release-minor   Bump minor, commit, and push (triggers release)"
	@echo "  release-major   Bump major, commit, and push (triggers release)"
	@echo ""
	@echo "  help            Show this help message"
