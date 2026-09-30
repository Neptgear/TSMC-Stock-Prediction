# Security policy

## Supported scope

This repository is a local research prototype, not a trading service. Security fixes are applied to the current `main` branch.

## Reporting

Do not publish API keys, private market-data credentials, personal files or exploit details in a public issue. Contact the repository owner through the GitHub profile and provide minimal reproduction steps without secrets.

## Safe use

- Store `ALPHAVANTAGE_API_KEY` and `FLASK_SECRET` in an untracked `.env` or environment variables.
- Bind the development server to loopback unless authentication, HTTPS and production hardening have been added.
- Treat downloaded market data and saved experiment output as third-party data subject to its provider's terms.
- This project does not execute trades and must not be presented as financial advice.
