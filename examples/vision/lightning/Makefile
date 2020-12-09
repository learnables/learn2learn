
.PHONY: *

proto-cifarfs:
	CUDA_VISIBLE_DEVICES=$(GPU) python main.py \
		--dataset=cifarfs \
		--max_epochs=1000 \
		--min_epochs=1000 \
		--algorithm=protonet \
		--lr=0.0016 \
		--meta_batch_size=16 \
		--train_shots=5 \
		--train_ways=15 \
		--train_queries=15 \
		--test_shots=5 \
		--test_ways=5 \
		--test_queries=5 \
		--seed=42

maml-cifarfs:
	CUDA_VISIBLE_DEVICES=$(GPU) python main.py \
		--dataset=cifarfs \
		--max_epochs=25000 \
		--min_epochs=25000 \
		--algorithm=maml \
		--adaptation_steps=5 \
		--adaptation_lr=0.05 \
		--lr=0.001 \
		--meta_batch_size=16 \
		--train_shots=5 \
		--train_ways=5 \
		--train_queries=5 \
		--test_shots=5 \
		--test_ways=5 \
		--test_queries=5 \
		--seed=42

anil-cifarfs:
	CUDA_VISIBLE_DEVICES=$(GPU) python main.py \
		--dataset=cifarfs \
		--max_epochs=2000 \
		--min_epochs=2000 \
		--algorithm=anil \
		--adaptation_steps=5 \
		--adaptation_lr=0.05 \
		--lr=0.001 \
		--meta_batch_size=16 \
		--train_shots=5 \
		--train_ways=5 \
		--train_queries=5 \
		--test_shots=5 \
		--test_ways=5 \
		--test_queries=5 \
		--seed=42

metaoptnet-cifarfs:
	CUDA_VISIBLE_DEVICES=$(GPU) python main.py \
		--dataset=cifarfs \
		--max_epochs=10000 \
		--min_epochs=10000 \
		--algorithm=metaoptnet \
		--train_shots=5 \
		--train_ways=5 \
		--train_queries=15 \
		--test_shots=5 \
		--test_ways=5 \
		--test_queries=5 \
		--seed=42

proto-mi:
	CUDA_VISIBLE_DEVICES=$(GPU) python main.py \
		--dataset='mini-imagenet' \
		--max_epochs=10000 \
		--min_epochs=10000 \
		--algorithm=protonet \
		--distance_metric='euclidean' \
		--meta_batch_size=8 \
		--lr=0.005 \
		--train_shots=5 \
		--train_ways=20 \
		--train_queries=15 \
		--test_shots=5 \
		--test_ways=5 \
		--test_queries=5 \
		--data_parallel \
		--seed=42

maml-mi:
	CUDA_VISIBLE_DEVICES=$(GPU) python main.py \
		--dataset=mini-imagenet \
		--max_epochs=35000 \
		--min_epochs=35000 \
		--algorithm=maml \
		--adaptation_steps=5 \
		--adaptation_lr=0.02 \
		--lr=0.0003 \
		--meta_batch_size=16 \
		--train_shots=5 \
		--train_ways=5 \
		--train_queries=5 \
		--test_shots=5 \
		--test_ways=5 \
		--test_queries=5 \
		--data_parallel \
		--seed=42

anil-mi:
	CUDA_VISIBLE_DEVICES=$(GPU) python main.py \
		--dataset=mini-imagenet \
		--lr=0.001 \
		--meta_batch_size=16 \
		--max_epochs=25000 \
		--min_epochs=25000 \
		--algorithm=anil \
		--adaptation_lr=0.1 \
		--adaptation_steps=5 \
		--train_shots=5 \
		--train_ways=5 \
		--train_queries=5 \
		--test_shots=5 \
		--test_ways=5 \
		--test_queries=5 \
		--seed=42

metaoptnet-mi:
	CUDA_VISIBLE_DEVICES=$(GPU) python main.py \
		--dataset=mini-imagenet \
		--lr=3e-4 \
		--max_epochs=40000 \
		--min_epochs=40000 \
		--algorithm=metaoptnet \
		--train_shots=15 \
		--train_ways=5 \
		--train_queries=5 \
		--test_shots=5 \
		--test_ways=5 \
		--test_queries=5 \
		--data_parallel \
		--seed=42
