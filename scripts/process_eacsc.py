"""
This script is to make the egyptian-arabic-conversational-speech-corpus comply with the same
structure of the MGB-3 data set. When you run this script, you expect it to output a CSV file
with 4 columns, [filename, sentence, start_time, end_time]
"""

import pandas as pd
import os
import numpy as np


def load_data(data_dir):
    ext = '.txt'
    data = pd.DataFrame()

    for file in os.listdir(data_dir):
        if file.endswith(ext):
            df = pd.read_csv(f"{data_dir}{file}", sep="\t",
                             header=None, names=["[start_time,end_time]", "speaker_id", "gender", "transcript"])
            df["filename"] = file.split(".")[0]
            data = pd.concat([data, df])
        else:
            continue
    return data


def parse_time_to_array(row):
    characters_to_remove = "[]"
    for character in characters_to_remove:
        row = row.replace(character, "")
    row = row.split(",")
    x = np.array(row)
    return x.astype(float)


def process_data(_data):
    # drop duplicates and unused columns and rename column to comply with MGB-3 naming
    _data = _data.drop_duplicates()
    _data = _data.drop(columns=["speaker_id", "gender"])
    _data = _data.rename(columns={"transcript": "sentence"})

    # parse time array from string to float
    _data["[start_time,end_time]"] = _data["[start_time,end_time]"].apply(parse_time_to_array)
    _data[["start_time", "end_time"]] = _data["[start_time,end_time]"].to_list()
    _data = _data.drop(columns=["[start_time,end_time]"])
    return _data


if __name__ == '__main__':
    data_dir = "./data/"
    eacsc_dir = "./EACSC/TXT/"

    data = load_data(eacsc_dir)
    data = process_data(data)
    data.to_csv(f"{data_dir}eacsc.csv", index=False)
