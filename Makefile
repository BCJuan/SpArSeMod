style:
	black --line-length 79 --target-version py37 examples sparsemod
	isort --recursive examples sparsemod

quality:
	black --line-length 79 --target-version py37 examples sparsemod
	isort --recursive examples sparsemod
	pylint sparsemod examples

PHONY: style quality
