SHELL := /bin/bash

.PHONY: download_dataset extract_dataset split_dataset dataset train train_places365 train_places365_small train_places10 train_places10_small

DATA_TAR := places365standard_easyformat.tar
DATA_URL := http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
DATA_DIR := data

dataset: download_dataset extract_dataset split_dataset
	@echo "Dataset ready."

download_dataset:
	@if [ -f "$(DATA_TAR)" ]; then \
		echo "Dataset tarball $(DATA_TAR) already exists. Skipping download."; \
	else \
		echo "Downloading dataset to ./$(DATA_TAR) ..."; \
		wget -c -O $(DATA_TAR) "$(DATA_URL)"; \
	fi

extract_dataset: $(DATA_TAR)
	@echo "Extracting dataset into $(DATA_DIR)/ ..."
	mkdir -p $(DATA_DIR)
	tar --skip-old-files -v -xf $(DATA_TAR) -C $(DATA_DIR)/

split_dataset:
	@if [ -f "$(DATA_DIR)/train.txt" ] && [ -f "$(DATA_DIR)/val.txt" ]; then \
		echo "Dataset splits already exist. Skipping split creation."; \
	else \
		echo "Creating dataset splits ..."; \
		uv run colorizer/utils/split_ds.py; \
	fi

train: train_places365

train_places365:
	uv run train.py \
		--device auto \
		--optimizer adam \
		--num-classes 365 \
		--batch-size 128 \
		--val-batch-size 128 \
		--epochs 15 \
		--train-data-path ./data/places365_standard/train \
		--val-data-path ./data/places365_standard/val \
		--compile \
		--wandb-log

train_places365_small:
	uv run train.py \
		--device auto \
		--optimizer adam \
		--num-classes 365 \
		--batch-size 128 \
		--val-batch-size 128 \
		--epochs 15 \
		--train-data-path ./data/places365_small/train \
		--val-data-path ./data/places365_small/val \
		--wandb-log

train_places10:
	uv run train.py \
		--device auto \
		--optimizer adam \
		--num-classes 10 \
		--batch-size 128 \
		--val-batch-size 128 \
		--epochs 15 \
		--train-data-path ./data/places10/train \
		--val-data-path ./data/places10/val \
		--compile \
		--wandb-log

train_places10_small:
	uv run train.py \
		--device auto \
		--optimizer adam \
		--num-classes 10 \
		--batch-size 128 \
		--val-batch-size 128 \
		--epochs 15 \
		--train-data-path ./data/places10_small/train \
		--val-data-path ./data/places10_small/val \
		--wandb-log
