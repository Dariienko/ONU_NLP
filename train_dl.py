import torch

from labsnlp.datasets import SequenceDataset
from labsnlp.models import TextClassificationModdule
from labsnlp.models import NaiveCustomLSTM

import numpy as np
import tqdm

from fire import Fire

def train(path_to_train_data, path_to_test_data, n_epochs=1, max_sequence_length=5, batch_size=16):
    # create Dataset and Dataloader
    train_dataset = SequenceDataset(
        path_to_data=path_to_train_data,
        max_sequence_length=max_sequence_length
    )
    test_dataset = SequenceDataset(
        path_to_data=path_to_test_data,
        max_sequence_length=max_sequence_length
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # collate_fn=SequenceDataset.collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        # collate_fn=SequenceDataset.collate_fn
    )

    # init model, loss, optimizer
    model = TextClassificationModdule(
        encoder=NaiveCustomLSTM(300, 300),
        classifier=torch.nn.Linear(300, 1) # torch.nn.Sequential(torch.nn.Linear(300, 150), torch.nn.BatchNorm1d(150), torch.nn.Dropout(0.25), torch....
    )

    loss = torch.nn.BCEWithLogitsLoss() #  log-sum-exp tric DYOR
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3) # see what's the difference between Adam & AdamW

    # tqdm -- DYOR
    for i in range(n_epochs):
        print(f"Epoch {i}")
        # Train loop
        train_tqdm = tqdm.tqdm(train_dataloader)
        model.train()
        for batch in train_tqdm:
            sentence, cls = batch['vectors'], batch['class']

            optimizer.zero_grad()
            model.zero_grad()

            output = model(sentence)

            L = loss(output.squeeze(1), cls.float())
            L.backward()


            optimizer.step()
            train_tqdm.set_description(f"Train loss: {L.item()}")
            train_tqdm.refresh()
            break

        # Validation loop
        model.eval()
        val_loss = []
        test_tqdm = tqdm.tqdm(test_dataloader)
        with torch.no_grad():
            for batch in test_tqdm:
                sentence, cls = batch['vectors'], batch['class']
                output = model(sentence)

                L = loss(output.squeeze(1), cls.float())
                val_loss.append(L.item())
                test_tqdm.set_description(f"Val loss: {L.item()}")
                test_tqdm.refresh()

        print(f"Epoch {i}, Val loss: {np.mean(val_loss)}")


if __name__ == "__main__":
    Fire(train)