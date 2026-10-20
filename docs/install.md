# Installation

Install the plugin from source:

```bash
git clone https://github.com/LLNL/aiida-flux-scheduler.git
cd aiida-flux-scheduler
pip install -e .
```

This installs the scheduler entry point named `flux`, which AiiDA can use when
configuring a computer.

## Documentation Dependencies

Install the documentation toolchain with:

```bash
pip install -e .[docs]
```

Build the HTML site locally:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html`.

## Development Extras

For local development, the available optional extras are:

- `.[docs]` for Sphinx, MyST, and generated reference pages
- `.[pre-commit]` for repository checks
- `.[tests]` for the project test dependencies
