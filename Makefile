SHELL=/bin/bash
LINT_PATHS=jaxfin/ tests/

pytest:
	python -m pytest --no-header -vv --html=test_report.html --self-contained-html

type: 
	mypy ${LINT_PATHS}

lint:
	ruff check jaxfin --output-format=full

complete-lint:
	lint
	pylint jaxfin --output-format=text:pylint_res.txt,colorized

format:
	isort ${LINT_PATHS}
	black ${LINT_PATHS}

check-codestyle:
	black ${LINT_PATHS} --check

commit-checks: format type lint

release: 
	python -m build
	twine upload dist/*

test-release: 
	python -m build
	twine upload dist/* -r testpypi

.PHONY: clean spelling doc lint format check-codestyle commit-checks