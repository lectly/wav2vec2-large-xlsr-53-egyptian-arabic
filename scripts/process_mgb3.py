import pandas as pd


def split(_string, sep, pos):
    _string = _string.split(sep)
    return [sep.join(_string[:pos]), sep.join(_string[pos:])]


def parse_text_lines(lines):
    # split name and time from speech
    lines = [line.split(" ", 1) for line in lines]

    # split name and time from each other
    name_time = [split(line[0], "_", 4) for line in lines]

    # split start time from end time
    time_split = [nt[1].split('_') for nt in name_time]

    for i in range(len(lines)):
        # remove old time
        del name_time[i][1]
        # add split time
        name_time[i].append(time_split[i][0])
        name_time[i].append(time_split[i][1])
        # add speech
        name_time[i].append(lines[i][1])

    return name_time


def load_data(data_dir):
    # load speech file name, times and speech
    with open(f"{data_dir}text_noverlap", encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    lines = parse_text_lines(lines)

    # load phonemes
    with open(f"{data_dir}text_noverlap.bw", encoding='utf-8', errors='ignore') as f:
        lines2 = f.readlines()
    lines2 = [line.split(" ", 1)[1] for line in lines2]

    assert len(lines) == len(lines2)
    for i in range(len(lines)):
        lines[i].append(lines2[i])
    return lines


if __name__ == '__main__':
    train_dir = "./mgb3/test/Omar/"
    dev_dir = "./mgb3/dev/Omar/"
    test_dir = "./mgb3/adapt/Omar/"
    columns = ["filename", "start_time", "end_time", "sentence", "phoneme"]
    data_dir = "./data/"

    lines = load_data(train_dir)
    train = pd.DataFrame(lines, columns=columns)
    train.to_csv(f"{data_dir}train.csv")

    lines = load_data(dev_dir)
    dev = pd.DataFrame(lines, columns=columns)
    dev.to_csv(f"{data_dir}dev.csv")

    lines = load_data(test_dir)
    test = pd.DataFrame(lines, columns=columns)
    test.to_csv(f"{data_dir}test.csv")
