
.PHONY: *

build:
	rm -f learn2learn/**/*.so
	rm -f learn2learn/**/*.c
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

predocs:
	#python scripts/compile_paper_list.py
	cp ./README.md docs/index.md
	cp ./CHANGELOG.md docs/changelog.md
	cp ./examples/vision/README.md docs/examples/vision.md
	cp ./examples/rl/README.md docs/examples/rl.md
	cp ./examples/optimization/README.md docs/examples/optim.md

docs: predocs
	mkdocs serve

docs-deploy: predocs
	mkdocs gh-deploy

# https://dev.to/neshaz/a-tutorial-for-tagging-releases-in-git-147e
release:
	echo 'Do not forget to bump the CHANGELOG.md'
	echo 'Tagging v'$(shell python -c 'print(open("learn2learn/_version.py").read()[15:-2])')
	sleep 3
	git tag -a v$(shell python -c 'print(open("learn2learn/_version.py").read()[15:-2])')
	git push origin --tags

publish:
	pip install -e .  # Full build
	rm -f learn2learn/*.so  # Remove .so files but leave .c files
	rm -f learn2learn/**/*.so
	python setup.py sdist  # Create package
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*  # Push to PyPI
