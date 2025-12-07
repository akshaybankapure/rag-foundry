# Contributing to RAG Foundry

We love your input! We want to make contributing to RAG Foundry as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## Development Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent.
4. You may merge the Pull Request in once you have the sign-off of two other developers.

## Code Style

We use `ruff` for linting and formatting. Please run `ruff check .` before submitting.

## Adding Plugins

Please see the "Extensibility" section in the README. When adding a new provider (e.g. a new Vector Store), please include a mock test case in `tests/` ensuring it instantiates correctly.
