TEST_PATH=./tests/
ROOT_PATH=./sparsemod/

clean-pycache:
	find . -type f -name "__pycache__" -exec rm -rf {} \;
	
clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

style:
	black --line-length 79 --target-version py37 examples sparsemod
	isort --recursive examples sparsemod

quality:
	black --line-length 79 --target-version py37 examples sparsemod
	isort --recursive examples sparsemod
	pylint sparsemod examples

test: clean-pycache
	python -m unittest discover $(TEST_PATH)

PHONY: style quality  clean-pycache
