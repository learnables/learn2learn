
pretrain-mi-res12:
	CUDA_VISIBLE_DEVICES=$(GPU) \
	MKL_NUM_THREADS=4 \
	OMP_NUM_THREADS=4 \
	python experiments/supervised_pretraining.py \
		--dataset='mini-imagenet' \
		--model='resnet12' \
		--iterations=150000 \
		--lr=0.01 \
		--weight_decay=0.0005 \
		--data_augmentation='jitter' \
		--batch_size=32 \
		--save_weights=1 \
		--seed=42

sp-ti-res12:
	CUDA_VISIBLE_DEVICES=$(GPU) \
	MKL_NUM_THREADS=4 \
	OMP_NUM_THREADS=4 \
	python experiments/supervised_pretraining.py \
		--dataset='tiered-imagenet' \
		--model='resnet12' \
		--iterations=150000 \
		--lr=0.01 \
		--weight_decay=0.0005 \
		--data_augmentation='jitter' \
		--batch_size=32 \
		--save_weights=1 \
		--seed=42

sp-cfs-cnn4:
	CUDA_VISIBLE_DEVICES=$(GPU) \
	MKL_NUM_THREADS=4 \
	OMP_NUM_THREADS=4 \
	python experiments/supervised_pretraining.py \
		--dataset='cifar-fs' \
		--model='cnn4' \
		--iterations=150000 \
		--lr=0.01 \
		--weight_decay=0.0005 \
		--data_augmentation='cut' \
		--batch_size=32 \
		--save_weights=1 \
		--seed=42

sp-fc100-cnn4:
	CUDA_VISIBLE_DEVICES=$(GPU) \
	MKL_NUM_THREADS=4 \
	OMP_NUM_THREADS=4 \
	python experiments/supervised_pretraining.py \
		--dataset='fc100' \
		--model='cnn4' \
		--iterations=150000 \
		--lr=0.01 \
		--weight_decay=0.0005 \
		--data_augmentation='cut' \
		--batch_size=32 \
		--save_weights=1 \
		--seed=42
