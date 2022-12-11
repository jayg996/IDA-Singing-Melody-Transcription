NOTE_LEVEL_CLASS = (
    [f"<time {i * 0.01:0.2f}>" for i in range(1025)]
    + [f"<pitch {i:d}>" for i in range(128)]
    + ["<offset>"]
    + ["<sos>", "<eos>", "<pad>"]
)
NOTE_LEVEL_DICT = {NOTE_LEVEL_CLASS[i]: i for i in range(len(NOTE_LEVEL_CLASS))}
