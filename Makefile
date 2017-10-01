.PHONY: all test single alt

all: model/*.py
	python .

test: test/*.py
	python -m unittest test.test_thresholds

single: test/*.py
	python -m unittest test.train_base.test_svc

