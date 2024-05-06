make_split_02:
	python scripts/train_dev_test_split.py --path_to_csv ./data/sarcasm_detection.csv --path_to_save ./data/ --test_size 0.3

train_dl_run:
	python train_dl.py D:\Code\ONU_NLP\data\train.csv  D:\Code\ONU_NLP\data\test.csv --n_epochs=1 --max_sequence_length=7 --batch_size=16