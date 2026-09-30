# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

A structured, self-teaching machine-learning / deep-learning / computer-vision
learning path — numbered folders of standalone scripts and notebooks meant to
be worked through roughly in order. It is not a packaged application or
library, and there is no build system, test suite, or CI. See `README.md` for
the full learning path with a per-file description table.

## Structure

Folders are numbered by learning stage; files within each folder are also
numbered to suggest an order:

- `01-python-numpy-basics/` — NumPy arrays, plain Python + scikit-learn intro.
- `02-data-visualisation/` — Matplotlib plotting basics.
- `03-scikit-learn-fundamentals/` — the classic load/split/train/evaluate
  workflow: KNN, decision trees, cross-validation, train/test-split
  evaluation, digit classification.
- `04-tensorflow-basics/` — legacy TF1 MNIST example (kept for reference,
  will not run as-is on modern TensorFlow — see the file's docstring).
- `05-pytorch-basics/` — tensors, autograd, a simple `nn.Module`, and a full
  training loop on the Iris dataset.
- `06-computer-vision-opencv/` — webcam capture, static-image face detection,
  real-time face detection, motion detection, and face-recognition training
  set prep. Includes its own `data/` (Haar cascade XML files) and `images/`
  (sample labeled face photos) assets.

## Documentation convention

Every notebook's **first cell is a markdown cell** explaining what the
notebook demonstrates; every script's **first statement is a module
docstring** doing the same. When adding a new file, add this header first —
it's what makes the repo usable as self-serve learning material. Match the
existing one-paragraph style (what it does + why it's here), not a full
tutorial in the header itself.

## Stack

Python 3, NumPy, pandas, Matplotlib/Seaborn, scikit-learn, TensorFlow,
PyTorch, OpenCV, Jupyter. Dependencies are listed (unpinned) in
`requirements.txt`; install with `pip install -r requirements.txt`.

## Working conventions

- Keep changes scoped to the file/notebook being discussed; these are
  independent exercises, not a shared codebase with common abstractions.
- New material goes in the folder matching its topic, numbered to continue
  that folder's existing sequence (or a new numbered folder for a new topic).
- When editing a notebook, prefer preserving existing cell outputs where the
  output isn't the thing being changed.
- Don't introduce project scaffolding (test frameworks, linters, packaging)
  unless explicitly asked — this repo is intentionally exploratory.
- Match the informal/experimental style already present (e.g. commented-out
  debug `print` calls) rather than imposing production coding standards,
  unless asked to clean a specific file up.
