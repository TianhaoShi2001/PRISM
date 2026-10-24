# PRISM

Released Criteo run of PRISM, with the training code, the split used for that run, and two checkpoints. Each checkpoint sits next to the `best_config.json` that built it. Evaluation reads that config by replacing the filename `best_model.pth` with `best_config.json`.

Run every command from the repository root.

## Setup

The released run used Python 3.9.21, PyTorch 2.5.1 with CUDA 12.1, NumPy 1.26.4, pandas 2.2.2, pyarrow 16.1.0, scikit-learn 1.5.2, and tqdm 4.67.1. The same pins are in `requirements.txt` and `VERSION`.

```bash
pip install -r requirements.txt
```

`--cuda` selects the visible GPU. `--cuda 0` is the default.

## 1. Prepare the data

Training and evaluation both read three parquet files. They are not in the repository. Build them from the public Criteo Uplift Prediction v2.1 file (13,979,592 rows). The model uses `f0`–`f11`, with `treatment` as the treatment, `visit` as the mediator, and `conversion` as the outcome.

`data/criteo_split_order.npy` is the row permutation for this run: 11,183,673 training rows, then 1,397,959 validation rows, then 1,397,960 test rows.

```bash
python data/prepare_criteo.py --raw /path/to/criteo-uplift-v2.1.csv
```

A `.csv.gz` path is accepted and unpacked first. If `--raw` is omitted, the script downloads the public archive and then applies the same split. The command writes:

```text
data/criteo/train.parquet
data/criteo/valid.parquet
data/criteo/test.parquet
```

## 2. Evaluate the released checkpoint

This loads the seed-42 outcome weights and scores the three splits. It does not train.

```bash
python main.py --mode eval --eval_ckpt checkpoints/criteo_seed42/best_model.pth --cuda 0
```

Metrics are written to `results/criteo/prism/final_metrics.json`. On the test split the seed-42 checkpoint gives AUUC 0.9145, Qini 0.4177, Lift@10 7.1703, and Lift@30 2.9908.

## 3. Train

The outcome config reads the mediator from `checkpoints/mediator/best_model.pth`. Those weights are already in the repository, so the outcome model can be trained without refitting the mediator.

```bash
python train_criteo.py --cuda 0 --seed 42
```

The same command is:

```bash
python main.py --mode reproduce --config_path checkpoints/criteo_seed42/best_config.json --seed 42 --cuda 0
```

`--seed 1042` and `--seed 2042` repeat the outcome run at the other two reported seeds. Seed 42 is the released checkpoint: among the three, its validation AUUC is the highest, and early stopping watches that number. Training stops after 10 epochs without an improvement, or at 100 epochs. A seed that already has `final_metrics.json` is skipped.

Outcome logs and metrics go to `results/criteo/prism/seed_<seed>/`.

To refit the mediator:

```bash
python train_mediator.py --cuda 0 --seed 42
```

```bash
python main.py --mode reproduce --config_path checkpoints/mediator/best_config.json --seed 42 --cuda 0
```

Do this before an outcome run only if you are replacing `checkpoints/mediator/best_model.pth`. The mediator output directory is the `run_hierarchy` in its config, with `seed_<seed>` appended.

The released outcome config uses batch size 262144 and four gradient-accumulation steps. Both hidden widths are `[128, 64, 32]`. The mediator uses weight decay `1e-6`. The outcome model uses weight decay `1e-5`, gate temperature 1, and sample-weight strengths 1, 1, and 1. Its config does not set `conflict_renorm`, so the training code rescales each batch by the mean weight. There is no search step.

## Checkpoints

| Weights | Config |
|---|---|
| `checkpoints/mediator/best_model.pth` | `checkpoints/mediator/best_config.json` |
| `checkpoints/criteo_seed42/best_model.pth` | `checkpoints/criteo_seed42/best_config.json` |

The mediator is the Stage I TARNet, seed 42. The outcome file is the seed-42 PRISM checkpoint.

## Metrics

Users are ranked by predicted uplift, `sigmoid(y1) - sigmoid(y0)`. AUUC is the area under the uplift curve after the curve is scaled to end at 1. Qini is the area under the scaled Qini curve, minus 1/2. Lift@10 and Lift@30 are the treated-minus-control conversion gap in that prefix, divided by the full-population Criteo ATE, +0.0012. Early stopping uses validation AUUC.

| Seed | Valid AUUC | Test AUUC | Test Qini | Test Lift@10 | Test Lift@30 |
|---|---:|---:|---:|---:|---:|
| 42 | 0.8940 | 0.9145 | 0.4177 | 7.1703 | 2.9908 |
| 1042 | 0.8844 | 0.9108 | 0.4139 | 7.0472 | 2.9668 |
| 2042 | 0.8923 | 0.9149 | 0.4173 | 7.1208 | 2.9114 |
| mean |  | 0.9134 | 0.4163 | 7.1128 | 2.9564 |
