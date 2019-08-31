
.PHONY: *

pro:
	python examples/rl/promp.py

dice:
	python examples/rl/maml_trpo_dice.py

publish:
	python setup.py sdist
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
