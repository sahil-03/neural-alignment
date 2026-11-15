# Do brain-based reward models resist overoptimization?

We compare an image-only reward model (ridge on DINOv2 features) with brain-based reward models
(DINOv2 → predicted fMRI responses → reward). Each is trained to predict a gold aesthetic score,
then used for best-of-n selection over COCO images that NSD never showed, to see how far each can
be pushed before its picks stop improving on the gold score.

- **Brain data:** Natural Scenes Dataset (NSD), 7T fMRI, 8 subjects × ~10k COCO images.
- **Candidates:** the 50,287 COCO 2017 images that are not in NSD.
- **Gold score:** LAION improved aesthetic predictor on CLIP ViT-L/14 embeddings.

## How it runs

Development happens locally. Heavy jobs run on [Modal](https://modal.com), and every derived
artifact is stored in a Cloudflare R2 bucket. Raw NSD data is streamed from NSD's public S3 bucket
and never copied; COCO lives on a Modal Volume.

```
core/
  config.py        settings (NA_* variables, read from the environment or core/.env)
  kit/r2/          R2 client
  storage/         Store: typed save/load over local disk or R2; artifact paths
  nsd/             NSD access, metadata, voxel masks, beta processing
  datasets/        data objects (features, betas, records, images, splits)
  pipelines/       local CLI for the data steps
  remote/          Modal app (modal_app.py) and remote entrypoints (data.py, ...)
```

Every step is idempotent: re-running skips finished work, so an interrupted run can simply be
started again.

## Reproducing

### 1. Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.12
- A Modal account
- A Cloudflare account with R2 enabled
- NSD access: NSD is publicly readable, but its terms ask you to submit the
  [NSD Data Access Agreement](https://naturalscenesdataset.org/) first.

```bash
uv sync
uv run modal token new        # skip if Modal is already set up on this machine
```

### 2. Storage: Cloudflare R2

1. Create an R2 bucket named `neural-alignment`, with the location hint
   **Eastern North America (ENAM)**. NSD's bucket and most Modal workers are in the US East.
2. Create an R2 API token (R2 → Manage API tokens) with **Object Read & Write** on that bucket.
   Copy the Access Key ID and Secret Access Key.
3. Copy `core/.env.example` to `core/.env` (gitignored) and fill it in:

   ```bash
   NA_DATA_ROOT=r2://neural-alignment/v1
   CLOUDFLARE_ACCOUNT_ID=...
   CLOUDFLARE_R2_ACCESS_KEY_ID=...
   CLOUDFLARE_R2_SECRET_ACCESS_KEY=...
   CLOUDFLARE_S3_ENDPOINT=https://<account_id>.r2.cloudflarestorage.com
   ```

### 3. Compute: Modal

Everything runs in a dedicated Modal environment named `neural-alignment`.

```bash
uv run modal environment create neural-alignment
make modal-secret             # creates the "r2" secret from core/.env in that environment
```

### 4. Data preparation

Run these in order. Steps 4a and 4b are independent and can run at the same time.

**4a. NSD: metadata, brain responses, splits.** Run once per subject (1–8):

```bash
uv run modal run -e neural-alignment core/remote/data.py::prepare --subject 1
```

It runs, in order:

| Step | What it does |
|---|---|
| `metadata` | builds the image table (73,000 NSD images) and per-subject trial tables from NSD's stimulus CSV |
| `verify` | checks that the voxel mask selects the same voxels from NSD's HDF5 and NIfTI betas (downloads ~1.5 GB) |
| `betas` | one container per session, in parallel: download the ~1 GB session file, keep the `nsdgeneral` voxels (~15k), z-score each voxel within the session |
| `aggregate` | stacks all sessions, averages repeated presentations of each image, and reports test-retest reliability |
| `splits` | test set = the shared1000 images the subject saw; the rest is train, in 5 fixed CV folds (seed 0) |

`metadata` and `verify` only need to run once, not per subject. To run only some steps:

```bash
uv run modal run -e neural-alignment core/remote/data.py::prepare --subject 2 --steps betas,aggregate,splits
```

**4b. COCO: images and candidate set.** Run once:

```bash
uv run modal run -e neural-alignment core/remote/data.py::coco
```

This downloads COCO 2017 train+val (~19 GB) into the Modal Volume `neural-alignment-coco`, then
writes the candidate table: all COCO 2017 images not in NSD (50,287).

### 5. Features, gold scores and models

*To be added.*

## Running the data steps locally

`core/pipelines/data.py` runs the same steps on your own machine against any store. This is
useful for small checks; the full beta processing downloads ~284 GB, so run that on Modal.

```bash
NA_DATA_ROOT=./data/store uv run python -m core.pipelines.data metadata
NA_DATA_ROOT=./data/store uv run python -m core.pipelines.data verify --subject 1
```

## Development

```bash
make lint-fix    # ruff
make check       # mypy
```
