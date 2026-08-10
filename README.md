# Intro to HPC Bootcamp: Teaching Students to Leverage LLMs for Regulatory Genomics on HPC

A hands-on educational project for introducing students to **genome language models (gLMs)**, DNA tokenization, Transformer models, DNABERT fine-tuning, and multi-GPU training on **NERSC Perlmutter**.

The project uses a binary regulatory-genomics task:

> **Given a DNA sequence, can a model distinguish a CTCF-binding region from background DNA?**

Students progress from exploring biological sequence data, to fine-tuning a pretrained DNA language model, to building a Transformer from scratch, and finally to scaling both approaches with GPUs and Slurm.

---

## Project Goals

By the end of the bootcamp, students should be able to:

- work with DNA sequences as machine-learning data;
- explain the biological motivation for predicting CTCF binding;
- explore genomic data with Python and `pandas`;
- compare different DNA tokenization strategies;
- explain the main parts of a Transformer;
- fine-tune a pretrained DNABERT model;
- build and train a custom Transformer from scratch;
- use GPUs on NERSC Perlmutter;
- understand basic Slurm concepts such as jobs, tasks, GPUs, ranks, and reservations;
- use PyTorch Distributed Data Parallel (DDP);
- compare model accuracy, AUROC, AUPRC, training time, GPU memory, and throughput.

The notebooks are intentionally written for students from mixed backgrounds. Some students may know biology but little programming, while others may know computer science but little genomics.

---

# Scientific Inspiration

This bootcamp was developed in part from ideas discussed in:

> **Liyuan Shu, Jiao Tang, Xiaoyu Guan, and Daoqiang Zhang.  
> "A comprehensive survey of genome language models in bioinformatics."  
> Briefings in Bioinformatics, Volume 27, Issue 1, 2026, bbaf724.**

The review discusses genome language-model architectures, sequence tokenization strategies, pretraining and fine-tuning, evaluation, downstream genomic applications, and the computational demands of genome language models.

- **Article / DOI:** https://doi.org/10.1093/bib/bbaf724
- **Open-access version (PubMed Central):** https://pmc.ncbi.nlm.nih.gov/articles/PMC12805252/

This repository is an **educational implementation inspired by the concepts in the review**. It is not an official implementation or reproduction of that paper.

---

# Notebook Roadmap

Run the notebooks in this order:

| Notebook | Main Question | What Students Do |
|---|---|---|
| **Notebook 0 — Preparing the Dataset** | How do we turn genomic coordinates into a machine-learning dataset? | Build positive CTCF examples and background sequences, perform QC, and write `seqs.txt` / `labels.txt`. |
| **Notebook 1 — Dataset + DNABERT** | Can a pretrained DNA language model learn CTCF binding? | Explore the dataset, learn 6-mer tokenization, inspect DNABERT, fine-tune it, evaluate metrics, and change training parameters. |
| **Notebook 2 — Tokenizers + Custom Transformer** | How does DNA representation affect a Transformer trained from scratch? | Implement multiple tokenization strategies, build attention and Transformer blocks, train models, and compare tokenizers. |
| **Notebook 3A — Maximum DNABERT Fine-Tuning** | How can we use HPC resources to fully fine-tune DNABERT? | Use BF16, Slurm, DDP, multiple GPUs, larger batches, full useful-parameter fine-tuning, and GPU-throughput measurements. |
| **Notebook 3B — Custom Transformer Across All Tokenizers** | How do the tokenizers compare when we scale the custom Transformer? | Train the larger custom Transformer with every tokenizer using multi-GPU DDP and compare scientific and computational results. |

Current student-facing filenames are:

```text
Notebook0_Preparing_Dataset_StudentReadable.ipynb
Notebook1_Dataset_DNABERT_StudentReadable.ipynb
Notebook2_Tokenizers_Transformer_StudentReadable.ipynb
Notebook3A_Maximum_DNABERT_StudentReadable_Bootcamp.ipynb
Notebook3B_Custom_Transformer_All_Tokenizers_StudentReadable.ipynb
```

---

# Designed for NERSC Perlmutter

The repository is designed primarily for **NERSC Perlmutter**.

NERSC provides JupyterLab on both login and compute nodes. For this bootcamp, a **Shared GPU Node** is convenient for the interactive Notebook 1 and Notebook 2 exercises, while Notebook 3A and Notebook 3B submit their larger GPU jobs through Slurm.

NERSC Jupyter documentation:

- https://docs.nersc.gov/services/jupyter/reference/
- https://docs.nersc.gov/beginner-guide/

NERSC recommends using Perlmutter Scratch for active, I/O-intensive computational work:

- https://docs.nersc.gov/filesystems/perlmutter-scratch/

> **Important:** Scratch is temporary storage and is subject to purging. Keep important code in GitHub and copy important final results to a persistent filesystem.

---

# Quick Start on Perlmutter

## 1. Start Jupyter at NERSC

Go to:

```text
https://jupyter.nersc.gov
```

Sign in with your NERSC account and MFA.

For the interactive parts of the bootcamp, select a **Shared GPU Node** unless your instructor tells you to use a different configuration.

Once JupyterLab opens, use the **Terminal** launcher.

---

## 2. Move to Perlmutter Scratch

In the Jupyter terminal:

```bash
cd $SCRATCH
```

You can confirm the location with:

```bash
pwd
```

NERSC defines `$SCRATCH` / `$PSCRATCH` automatically for each user.

---

## 3. Clone the GitHub repository


```bash
git clone https://github.com/joserico00/Intro-to-HPC-Bootcamp-Teaching-Students-to-Leverage-LLMs-for-Regulatory-Genomics-on-HPC.git

For example:

```bash
git clone https://github.com/joserico00/Intro-to-HPC-Bootcamp-Teaching-Students-to-Leverage-LLMs-for-Regulatory-Genomics-on-HPC.git
```

Then enter the repository:

```bash
cd Intro-to-HPC-Bootcamp-Teaching-Students-to-Leverage-LLMs-for-Regulatory-Genomics-on-HPC
```

Optional check:

```bash
ls
```

You should see the bootcamp notebooks.

---

## 4. Open the cloned folder in JupyterLab

Return to the JupyterLab file browser and navigate to the repository you cloned under `$SCRATCH`.

Open the notebooks from that repository directory.

The notebooks use the current working directory as the project directory by default:

```python
PROJECT_DIR = Path(
    os.environ.get("DNA_BOOTCAMP_HOME", ".")
).expanduser().resolve()
```

Therefore, if you open the notebooks from the cloned repository directory, results and generated scripts will stay with that project.

---

# Perlmutter Users: Skip Notebook 0

If you are using the current bootcamp allocation on Perlmutter, **you do not need to run Notebook 0**.

The prepared CTCF/K562 dataset already exists in project storage:

```text
/global/cfs/cdirs/m4388/projects/project7/ctcf_k562_example
```

The directory contains the files used by the later notebooks:

```text
seqs.txt
labels.txt
```

Therefore, the normal Perlmutter bootcamp order is:

```text
Notebook 1
   ↓
Notebook 2
   ↓
Notebook 3A
   ↓
Notebook 3B
```

Notebook 0 is included so students can understand how a genomic dataset can be constructed and so the workflow can be reproduced on another system or with another transcription factor.

> **Project/account note:** The current notebooks use NERSC project **`m4388`**. If your allocation or project directory is different, update the paths described in the configuration section below.

---

# Dataset

The current task uses:

- **organism:** human;
- **reference assembly:** hg38 / GRCh38;
- **cell line:** K562;
- **target:** CTCF;
- **positive class (`1`):** CTCF-binding DNA;
- **background class (`0`):** background DNA;
- **sequence length:** approximately 200 bp.

The prepared machine-learning dataset is represented by two simple files:

```text
seqs.txt
labels.txt
```

Each line in `seqs.txt` corresponds to the label on the same line in `labels.txt`.

Example:

```text
seqs.txt
ACGTACGT...
TTGCAACC...
...

labels.txt
1
0
...
```

---

# Notebook 0: Rebuilding the Dataset

Run Notebook 0 if:

- you are **not** using the prepared Perlmutter dataset;
- you want to understand how the dataset was generated;
- you want to use a different transcription factor, cell line, genome assembly, or sequence window.

The current Notebook 0 configuration points to:

```text
PEAKS_FILE
/global/cfs/cdirs/m4388/projects/project7/data/peaks/ENCFF519CXF.bed.gz

GENOME_FILE
/global/cfs/cdirs/m4388/projects/project7/data/genome/hg38.fa.gz

BLACKLIST_FILE
/global/cfs/cdirs/m4388/projects/project7/data/peaks/ENCFF356LFX.bed.gz
```

Important variables in Notebook 0 include:

```python
DATASET_NAME = "ctcf_k562"
PEAKS_FILE = "..."
GENOME_FILE = "..."
BLACKLIST_FILE = "..."

WINDOW_SIZE = 200
RANDOM_NEG_MULTIPLIER = 1
SEED = 42
```

Notebook 0 uses `pybedtools`, so a non-NERSC environment also needs a working **BEDTools** installation in addition to the Python package.

---

# Python Environment

The notebooks require a Python environment with packages including:

```text
numpy
pandas
matplotlib
scikit-learn
torch
transformers
huggingface_hub
tqdm
pybedtools        # Notebook 0
```

Notebook 3A is able to check likely training environments before submitting the Slurm job.

The current Perlmutter project environment includes:

```text
/global/cfs/cdirs/m4388/envs/dna-llm/bin/python
```

A quick environment check from a terminal is:

```bash
/global/cfs/cdirs/m4388/envs/dna-llm/bin/python -c \
"import torch, transformers, sklearn, pandas, numpy; print(torch.__version__)"
```

If this succeeds, the main machine-learning dependencies are available.

---

# DNABERT Model

The DNABERT notebooks use:

```text
zhihan1996/DNA_bert_6
```

The model expects DNA represented as overlapping **6-mers**.

For a 200-bp sequence:

```text
200 bases
   ↓
195 overlapping 6-mers
   ↓
special tokens
   ↓
DNABERT
```

Notebook 1 introduces this workflow.

Notebook 3A later performs a larger full-fine-tuning run.

---

# Tokenization Methods

Notebook 2 and Notebook 3B examine five representations:

| Tokenizer | Basic Idea |
|---|---|
| `single_nucleotide` | Each A, C, G, or T is one token. |
| `one_hot` | Each nucleotide is represented as a four-value vector. |
| `overlap_6mer` | Sliding 6-base windows overlap by five bases. |
| `nonoverlap_6mer` | DNA is divided into separate 6-base chunks. |
| `bpe` | Byte Pair Encoding learns frequently occurring sequence chunks from the training data. |

Notebook 3B trains the larger custom Transformer once for **each tokenizer**.

---

# Running Notebook 1

Notebook 1 performs the first major model-training exercise.

Main workflow:

```text
load DNA
↓
inspect and clean data
↓
convert DNA to overlapping 6-mers
↓
load pretrained DNABERT
↓
fine-tune
↓
calculate metrics
↓
graph results
↓
change one parameter
↓
compare runs
```

If running on Perlmutter, confirm:

```python
DATA_DIR = "/global/cfs/cdirs/m4388/projects/project7/ctcf_k562_example"
```

and verify that the notebook reports a CUDA device.

Example:

```text
Device : cuda
GPU    : NVIDIA A100...
```

---

# Running Notebook 2

Notebook 2 constructs the custom Transformer rather than relying on DNABERT.

Students implement or inspect:

```text
tokenization
embeddings
self-attention
multi-head attention
feed-forward network
residual connections
layer normalization
Transformer blocks
classification
```

For the cleanest comparison, the same cleaned dataset and train/validation split are used across representations.

---

# Running Notebook 3A

Notebook 3A is the HPC/full-fine-tuning DNABERT notebook.

It introduces:

- Slurm;
- NERSC GPU allocations;
- PyTorch Distributed Data Parallel;
- ranks and world size;
- BF16 automatic mixed precision;
- multi-GPU training;
- larger local/global batches;
- examples per second;
- peak GPU memory;
- full useful-parameter fine-tuning.

## Important variables

Near the beginning of Notebook 3A:

```python
NERSC_ACCOUNT = "m4388"

RUN_MODE = "bootcamp"
BOOTCAMP_DAY = 1

GPU_COUNT = 4
LOCAL_BATCH_SIZE = 32
```

### During the bootcamp

Use:

```python
RUN_MODE = "bootcamp"
```

and select the correct:

```python
BOOTCAMP_DAY
```

The notebook contains a `BOOTCAMP_RESERVATIONS` dictionary.

**Reservation names and times are bootcamp-specific.** Verify them before the event and update them if the NERSC reservation changes.

### Outside bootcamp hours

For ordinary small tests on Perlmutter:

```python
RUN_MODE = "shared"
GPU_COUNT = 1
```

or:

```python
RUN_MODE = "shared"
GPU_COUNT = 2
```

Current NERSC policy allows GPU jobs in the `shared` QOS to request 1 or 2 GPUs.

See:

https://docs.nersc.gov/systems/perlmutter/running-jobs/

---

# Running Notebook 3B

Notebook 3B uses the custom Transformer from Notebook 2 at a larger scale.

The default model configuration is intentionally larger than the introductory Notebook 2 model:

```python
MODEL_CONFIG = {
    "n_embd": 256,
    "n_head": 8,
    "n_layer": 8,
    "dropout": 0.10,
    "epochs": 10,
    "local_batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "precision": "bf16",
    "bpe_merges": 80,
    "seed": 42,
}
```

It trains the model with:

```text
single nucleotide
one-hot
overlapping 6-mer
non-overlapping 6-mer
BPE
```

and saves a combined comparison.

For the final DNABERT-vs-custom-model comparison, run **Notebook 3A before Notebook 3B**.

---

# Slurm and GPU Configuration on Perlmutter

Notebook 3A and Notebook 3B generate Slurm scripts automatically.

Important NERSC concepts include:

```text
-A / --account        project allocation
-C gpu                request GPU nodes
-q                    QOS / queue
-N                    number of nodes
--gpus-per-node       GPUs requested on each node
--ntasks-per-node     DDP processes on each node
--gpu-bind            task/GPU visibility
```

The notebooks map Slurm tasks to PyTorch DDP processes:

```text
SLURM_PROCID  → RANK
SLURM_LOCALID → LOCAL_RANK
SLURM_NTASKS  → WORLD_SIZE
```

NERSC Perlmutter job documentation:

https://docs.nersc.gov/systems/perlmutter/running-jobs/

---

# If You Are NOT Running on Perlmutter

Notebook 0, Notebook 1, and Notebook 2 can be adapted to another Linux workstation, HPC system, or compatible GPU environment.

Notebook 3A and Notebook 3B are more tightly coupled to **NERSC + Slurm + NVIDIA GPUs** and require additional changes on another system.

## Variables and paths to review

### All notebooks

The notebooks support:

```text
DNA_BOOTCAMP_HOME
```

to choose the project/output directory.

If it is not set, the notebooks use the directory from which they are opened.

---

### Notebook 0

Change:

```python
PEAKS_FILE
GENOME_FILE
BLACKLIST_FILE
DATASET_NAME
```

You may also change:

```python
WINDOW_SIZE
RANDOM_NEG_MULTIPLIER
SEED
```

---

### Notebook 1

Change:

```python
DATA_DIR
```

to the directory containing:

```text
seqs.txt
labels.txt
```

The notebook automatically uses CUDA if `torch.cuda.is_available()` is true.

---

### Notebook 2

Change:

```python
DATA_DIR
```

to your dataset directory.

The custom Transformer also automatically chooses CUDA when available.

---

### Notebook 3A

Review or replace:

```python
DATA_DIR
NERSC_ACCOUNT
RUN_MODE
BOOTCAMP_DAY
BOOTCAMP_RESERVATIONS
GPU_COUNT
LOCAL_BATCH_SIZE
PYTHON_CANDIDATES
```

The current shared environment path:

```text
/global/cfs/cdirs/m4388/envs/dna-llm/bin/python
```

is Perlmutter-specific.

If you are on another Slurm cluster, you must also review the generated `#SBATCH` options.

If you are not using Slurm, the launcher section will need to be replaced with the job-launch mechanism used by your system.

---

### Notebook 3B

Review or replace:

```python
DATA_DIR
NERSC_ACCOUNT
RUN_MODE
BOOTCAMP_DAY
BOOTCAMP_RESERVATIONS
GPU_COUNT
LOCAL_BATCH_SIZE
NOTEBOOK_PYTHON
```

Also review the generated Slurm options and DDP launch command for your HPC environment.

---

# Recommended Directory Layout

After running the notebooks, the repository may look similar to:

```text
project/
│
├── Notebook0_Preparing_Dataset_StudentReadable.ipynb
├── Notebook1_Dataset_DNABERT_StudentReadable.ipynb
├── Notebook2_Tokenizers_Transformer_StudentReadable.ipynb
├── Notebook3A_Maximum_DNABERT_StudentReadable_Bootcamp.ipynb
├── Notebook3B_Custom_Transformer_All_Tokenizers_StudentReadable.ipynb
│
├── notebook1_results/
├── notebook2_results/
├── notebook3a_results/
├── notebook3a_scripts/
├── notebook3b_results/
└── notebook3b_scripts/
```

The generated result directories may contain:

```text
*_history.csv
*_predictions.csv
*_summary.json
*_parameters.csv
*_parameter_groups.csv
*_checkpoint.pt
Slurm output logs
```

---

# Metrics Used

Students encounter several model-evaluation metrics:

- **Accuracy** — fraction of predictions that are correct.
- **Precision** — among predicted binding sequences, how many are truly binding?
- **Recall / Sensitivity** — among real binding sequences, how many did the model find?
- **Specificity** — among background sequences, how many did the model correctly reject?
- **F1 score** — balance between precision and recall.
- **AUROC** — how well the model ranks positive examples above negative examples across thresholds.
- **AUPRC** — precision-recall performance across thresholds.

Computational measurements include:

- training time;
- time per epoch;
- examples per second;
- number of trainable parameters;
- GPU count;
- peak GPU memory;
- parallel speedup and efficiency where applicable.

---

# Reproducibility

The notebooks use fixed random seeds where possible:

```python
SEED = 42
```

For controlled comparisons, students should generally:

1. use the same cleaned dataset;
2. keep the train/validation split fixed;
3. keep the seed fixed;
4. change **one major variable at a time**;
5. record the configuration with the result.

GPU/distributed training can still contain sources of nondeterminism, so identical seeds do not guarantee bit-for-bit identical results.

---

# Common Problems

## `FileNotFoundError` for `seqs.txt` or `labels.txt`

Check:

```python
DATA_DIR
```

On the current Perlmutter project it should point to:

```text
/global/cfs/cdirs/m4388/projects/project7/ctcf_k562_example
```

---

## `ModuleNotFoundError: No module named 'torch'`

The batch job is using the wrong Python environment.

On the current Perlmutter project, check:

```text
/global/cfs/cdirs/m4388/envs/dna-llm/bin/python
```

Notebook 3A includes a Python-environment preflight before submitting the training job.

---

## Slurm rejects the GPU request

Check:

```python
NERSC_ACCOUNT
RUN_MODE
GPU_COUNT
BOOTCAMP_DAY
BOOTCAMP_RESERVATIONS
```

Also verify that the reservation is currently active if using:

```python
RUN_MODE = "bootcamp"
```

---

## Running outside the bootcamp reservation

Use:

```python
RUN_MODE = "shared"
```

and request only:

```python
GPU_COUNT = 1
```

or:

```python
GPU_COUNT = 2
```

for small Perlmutter tests.

---

## Hugging Face model download problems

DNABERT is loaded from Hugging Face when it is not already cached.

Check that the compute environment can access the model or that it has already been cached.

The model identifier used here is:

```text
zhihan1996/DNA_bert_6
```

---

# Suggested Bootcamp Progression

A possible teaching sequence is:

```text
Biology + Python
      ↓
Explore CTCF dataset
      ↓
DNABERT tokenization
      ↓
Fine-tune pretrained DNABERT
      ↓
Build self-attention
      ↓
Build custom Transformer
      ↓
Compare DNA tokenization strategies
      ↓
Learn Slurm + DDP
      ↓
Scale full DNABERT fine-tuning
      ↓
Scale custom Transformer across tokenizers
      ↓
Compare biological performance and compute cost
```

The emphasis is not only on obtaining the highest score.

Students should be able to explain:

> **What representation did we use? What model did we train? What computational resources did we use? What did the result tell us biologically and computationally?**

---

# References

## Genome language models

Shu L, Tang J, Guan X, Zhang D. **A comprehensive survey of genome language models in bioinformatics.** *Briefings in Bioinformatics*. 2026;27(1):bbaf724.

DOI:

https://doi.org/10.1093/bib/bbaf724

Open-access article:

https://pmc.ncbi.nlm.nih.gov/articles/PMC12805252/

## NERSC / Perlmutter

NERSC Jupyter documentation:

https://docs.nersc.gov/services/jupyter/reference/

NERSC beginner guide:

https://docs.nersc.gov/beginner-guide/

Perlmutter Scratch:

https://docs.nersc.gov/filesystems/perlmutter-scratch/

Running GPU jobs on Perlmutter:

https://docs.nersc.gov/systems/perlmutter/running-jobs/

NERSC queue/QOS policy:

https://docs.nersc.gov/jobs/policy/

---

# Acknowledgment

This project is designed for educational use on the **NERSC Perlmutter** supercomputer and demonstrates how HPC resources can be incorporated into genome-language-model training workflows.

---

## Repository Setup Checklist

Before the bootcamp:

```text
[ ] GitHub repository is cloned into $SCRATCH
[ ] Notebooks open from the cloned repository
[ ] Python environment imports torch and transformers
[ ] Prepared dataset path is accessible
[ ] NERSC_ACCOUNT is correct
[ ] Bootcamp reservation names/dates are current
[ ] RUN_MODE is set correctly
[ ] GPU_COUNT is appropriate for the selected QOS/reservation
[ ] Notebook 1 can see a GPU
[ ] Notebook 3A/3B Slurm smoke/preflight checks succeed
```
