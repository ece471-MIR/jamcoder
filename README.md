# jamcoder
majical voice jamcoder

### uv Virtual Environment and Dependency Installation
Install [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) before continuing.
```bash
uv venv
uv pip install -r requirements.txt
```

### Prerequisite: Phoneme Dataset
See [data/README.md](data/README.md)

### Usage

```
[jamcoder]$ uv run python src/jamcoder.py --help
usage: jamcoder [-h] [-nc] [-w CROSSFADE_OVERLAP] [-nd] [-o OUTFILE] -u VOICE -s SENTENCE [SENTENCE ...] [-d]

options:
  -h, --help            show this help message and exit
  -nc, --no-crossfade
  -w, --crossfade-overlap CROSSFADE_OVERLAP
                        Sets the overlap of the crossfade, from 0.0 to 1.0. Higher means more overlap.
  -nd, --no-dual-similarity
  -o, --outfile OUTFILE
  -u, --voice VOICE
  -s, --sentence SENTENCE [SENTENCE ...]
  -d, --debug
```

#### Generating a sample

Lets say you have a dataset of `TextGrid`s and `wav`s stored at `data/james`.

You can have this `james` voice synthesize a (plausible) sentence with:
```
$ uv run python src/jamcoder.py -u james -s hey whats up
```

This will generate synthesized speech to `synth.wav` in your present working directory.

