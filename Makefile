.PHONY: test lint benchmark compare api

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

lint:
	ruff check src tests

benchmark:
	PYTHONPATH=src python -m hybrid_scheduler benchmark

compare:
	PYTHONPATH=src python -m hybrid_scheduler compare results/benchmark.csv greedy_local_search warm_start_permutation_qaoa_p1

api:
	uvicorn hybrid_scheduler.api:app --host 0.0.0.0 --port 8000

