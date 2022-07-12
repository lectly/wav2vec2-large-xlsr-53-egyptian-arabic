import pandas as pd
from datasets import load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.utils.data import Dataset


class MGB3Dataset(Dataset):
    def __init__(self, df):
        super(Dataset, self).__init__()
        self.data = df[["input_values", "input_length", "labels"]].to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


test = pd.read_pickle(f"./data/train.pbz2", compression='bz2')
test_dataset = MGB3Dataset(test)


wer = load_metric("wer")

processor = Wav2Vec2Processor.from_pretrained("arbml/wav2vec2-large-xlsr-53-arabic-egyptian")
model = Wav2Vec2ForCTC.from_pretrained("arbml/wav2vec2-large-xlsr-53-arabic-egyptian")
model.to("cuda")

print("WER: {:2f}".format(100 * wer.compute(predictions=test_dataset["pred_strings"],
                                            references=test_dataset["sentence"])))
