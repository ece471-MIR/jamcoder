### Phoneme Data
Organise the data directory as shown below:
```
data/
├── voice_1/
│   ├── bed.TextGrid
│   ├── bed.wav
│   │   ...
│   └── zoo.wav
├── voice_2/
│   ...
└── voice_n/
```

Files within voice directories that do not belong to a `wav`-`TextGrid` pair are ignored.

For each voice and every `WORD` in the phonemic chart below[^1], maintain a `WORD.wav` and corresponding `WORD.TextGrid` file in the voice's subdirectory, where each `TextGrid` contains orthographic and phonemic interval annotations.

![Phonemic Chart](phonemic-chart.jpg)

[^1]: “Phonemic Chart,” EnglishClub, https://www.englishclub.com/pronunciation/phonemic-chart.php (accessed Nov. 30, 2025).