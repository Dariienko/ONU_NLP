make_split_02:
	python scripts/train_test_split.py --path_to_csv ./data/sarcasm_detection.csv --path_to_save ./data/ --test_size 0.2