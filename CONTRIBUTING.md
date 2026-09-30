# Contributing

Contributions should make the experiment easier to reproduce or audit, not merely improve a selected chart.

1. Create a branch from the latest `main`.
2. Install `requirements.txt` in an isolated Python 3.12 environment.
3. Run `python -m unittest discover -s tests -v` and `python scripts/verify_portfolio.py`.
4. Do not commit `.env`, API keys, caches, logs, large downloaded datasets or unverifiable model claims.
5. Document the cutoff date, split rule, feature availability, random seeds, baseline and all failed runs relevant to a result.

Any claim that a model outperforms a baseline must include repeated out-of-sample evaluation and uncertainty. A single favorable period is not sufficient evidence.
