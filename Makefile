
.PHONY: *

dist-promp:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -m torch.distributed.launch \
	          --nproc_per_node=4 \
		  examples/rl/dist_promp.py

promp:
	python examples/rl/promp.py

dice:
	python examples/rl/maml_dice.py

publish:
	python setup.py sdist
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
