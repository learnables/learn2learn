
.PHONY: *

build:
	python setup.py build_ext --inplace

clean:
	rm -rf dist/ build/
	rm -f learn2learn/**/*.c
	rm -f learn2learn/**/*.so
	rm -f learn2learn/**/*.html
	rm -f learn2learn.egg-info/**/*.html

# Admin
dev:
	pip install --progress-bar off torch gym pycodestyle >> log_install.txt
	python setup.py develop

lint:
	pycodestyle learn2learn/ --max-line-length=160

lint-examples:
	pycodestyle examples/ --max-line-length=80

lint-tests:
	pycodestyle tests/ --max-line-length=180

tests:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -W ignore -m unittest discover -s 'tests' -p '*_test.py' -v
	make lint

notravis-tests:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -W ignore -m unittest discover -s 'tests' -p '*_test_notravis.py' -v

alltests: 
	rm -f alltests.txt
	make tests >>alltests.txt 2>&1
	make notravis-tests >>alltests.txt 2>&1

docs:
	cd docs && pydocmd build && pydocmd serve

docs-deploy:
	cd docs && pydocmd gh-deploy

publish:
	python setup.py sdist
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
