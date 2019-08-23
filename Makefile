
.PHONY: *

omni:
	python examples/vision/maml_omniglot.py

publish:
	python setup.py sdist
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
