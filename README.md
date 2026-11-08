# world-models-lab

System identification and rollout evaluation.

## Scope

A small, inspectable, reproducible **v0.1 reference implementation**. The project is not presented as a production system or peer-reviewed research result.

## Requirements

Python 3.12; NumPy

## Setup

```bash
python -m pip install -r requirements-research.txt
```

## Run example / experiment

From the repository root:

```bash
python -m experiments.synthetic_rollout
```

## Test

```bash
python -m pytest -q
```

## Reproducibility

Experiments use deterministic inputs or explicitly seeded generators. See the experiment sources for assumptions and input sizes. CI exercises both tests and the example; check the Actions tab for actual run results. No empirical results are claimed without executed measurements.

## Limitations

This is a linear state-space system-identification baseline. States are directly observed; no neural latent encoder, learned visual representation, model-based controller or real-world validation is included.

## License

MIT. Contributions should include a focused regression test and an explanation of changed assumptions.
