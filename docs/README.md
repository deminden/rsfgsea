# Documentation

This project has three user-facing entrypoints:

- Rust library: use the `fgsea*` APIs from `rsfgsea`
- CLI: run preranked enrichment from files
- Python bindings: call the same fgsea-compatible workflows from Python

Start here:

- [CLI Guide](./cli.md)
- [Python Guide](./python.md)
- [Plotting Guide](./plotting.md)
- [Algorithm Guide](./algorithms.md)
- [Development Guide](./development.md)
- [Reproducibility Guide](./reproducibility.md)

Suggested reading order:

1. Read the [CLI Guide](./cli.md) or [Python Guide](./python.md) depending on how you plan to run the tool.
2. Read the [Plotting Guide](./plotting.md) if you need enrichment-plot output from CLI, Python, or R.
3. Read the [Algorithm Guide](./algorithms.md) if you need to understand mode selection, score semantics, or GPU behavior.
4. Read the [Reproducibility Guide](./reproducibility.md) for parity workflows, scripts, and reference data generation.
5. Read the [Development Guide](./development.md) if you are changing code, tests, or release metadata.
