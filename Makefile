
.PHONY: *

publish:
	python setup.py sdist
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
