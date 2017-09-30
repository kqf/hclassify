.PHONY: all test

all: model/*.py
	python .

test: test/*.py
	python -m unittest test.test_thresholds

single: test/*.py
	python -m unittest test.test_single
