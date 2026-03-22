# Contributing

Thanks for contributing to `identity-risk-engine`.

## Setup
```bash
python3.11 -m pip install -e .[dev]
```

## Validation
```bash
make lint PYTHON=python3.11
make test PYTHON=python3.11
```

## Pull Requests
- Keep changes focused and include tests.
- Add/update docs for API and behavior changes.
- Do not commit secrets or real user data.
- Prefer synthetic examples for demos and notebooks.
