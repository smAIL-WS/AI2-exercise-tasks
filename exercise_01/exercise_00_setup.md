# Exercise 00: Setting Up Your Development Environment

## Overview

This document walks you through the one-time setup of your development environment for this course. Follow each section in order. By the end, you will have VS Code installed, an SSH connection to the GPU server, the shared Python environment activated, and an understanding of how to submit GPU jobs.

**Estimated time:** 30–45 minutes

---

## 1. Visual Studio Code

### What is VS Code?

Visual Studio Code (VS Code) is a free, lightweight code editor. It supports Python out of the box, integrates with Git, and — most importantly for this course — allows you to connect directly to the GPU server and edit files there as if they were on your local machine.

### 1.1 Installation

1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/).
2. Run the installer for your operating system (Windows, macOS, or Linux).
3. Launch VS Code after installation.

### 1.2 Essential Extensions

Open VS Code and go to the Extensions panel (click the square icon on the left sidebar, or press `Ctrl+Shift+X` / `Cmd+Shift+X`). Search for and install the following extensions:

**Required:**

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| Python | Microsoft | Python language support, linting, debugging |
| Pylance | Microsoft | Fast, accurate Python IntelliSense and type checking |
| Remote - SSH | Microsoft | Connect to the GPU server and work remotely |
| Jupyter | Microsoft | View and run Jupyter notebooks if needed |

**Recommended:**

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| GitLens | GitKraken | Enhanced Git integration — view blame, history, and diffs inline |
| Python Indent | Kevin Rose | Corrects Python indentation automatically |
| autoDocstring | Nils Werner | Generate Python docstrings with a shortcut |
| YAML | Red Hat | Syntax support for YAML config files |

### 1.3 Configuring Python in VS Code

After installing the Python extension:

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).
2. Type `Python: Select Interpreter` and press Enter.
3. Select the Python interpreter you want to use. On the GPU server (after connecting via SSH), this will be the shared conda environment provided for the course.

---

## 2. SSH Connection to the GPU Server

All GPU-dependent exercises run on the course server, not on your local machine. You will connect to this server from VS Code using SSH.

### 2.1 What is SSH?

SSH (Secure Shell) is a protocol that lets you securely access a remote machine's terminal and file system from your local computer. Once connected, VS Code behaves as if the remote server's files are local — you can edit, run, and debug code directly.

### 2.2 Obtain Your Credentials

Your instructor will provide you with:

- **Server address** (hostname or IP)
- **Username**
- **Password** (or an SSH key file)

Keep these credentials private. Do not share them with other students.

### 2.3 Connect via VS Code

1. Open VS Code.
2. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).
3. Type `Remote-SSH: Connect to Host...` and select it.
4. Enter the connection string in the following format:

```
your_username@server_address
```

5. If prompted, select `Linux` as the remote platform.
6. Enter your password when asked.
7. VS Code will install its server component on the remote machine (this happens only on first connection and takes a minute or two).

Once connected, the bottom-left corner of VS Code will show `SSH: <server_address>`, confirming you are working on the remote server.

### 2.4 Setting Up SSH Keys (Recommended)

Typing your password every time you connect is inconvenient. SSH keys let you authenticate automatically.

**On your local machine**, open a terminal and run:

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```

Press Enter to accept the default file location. You may set a passphrase or leave it empty.

Then copy your public key to the server:

- **macOS/Linux:**

```bash
ssh-copy-id your_username@server_address
```

- **Windows (PowerShell):**

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh your_username@server_address "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

After this, VS Code will connect without asking for a password.

### 2.5 Opening a Remote Folder

After connecting via SSH:

1. Click `File → Open Folder` (or `Ctrl+K Ctrl+O`).
2. Navigate to your home directory on the server (e.g. `/home/your_username/`).
3. Select the folder you want to work in and click OK.

You can now browse, edit, and create files on the server directly from VS Code's file explorer.

### 2.6 Using the Integrated Terminal

Open a terminal within VS Code via `Terminal → New Terminal` (or `` Ctrl+` ``). This terminal runs on the remote server, not your local machine. You will use it to run Git commands, activate conda environments, submit Slurm jobs, and execute scripts.

---

## 3. Activating the Course Environment

After connecting to the server via SSH, you need to activate the shared Python environment before running any code. This environment contains all the packages required for the course (PyTorch, torchvision, NumPy, matplotlib, TensorBoard, etc.).

### 3.1 First-Time Activation

The first time you open a terminal after connecting, run these two commands:

```bash
source /etc/profile.d/course_env.sh
activate_course
```

The first command loads the activation shortcut. The second activates the environment. Your terminal prompt will change to:

```
(course_env) your_username@server:~$
```

This confirms the environment is active.

### 3.2 On Subsequent Logins

After the first time, the shortcut is loaded automatically when you open a new terminal. You only need to run:

```bash
activate_course
```

If you see `activate_course: command not found`, run `source /etc/profile.d/course_env.sh` first, then `activate_course`.

### 3.3 Verify the Environment

After activating, confirm everything works:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:

```
PyTorch: 2.6.0+cu124
CUDA: True
```

If `CUDA: False` appears, contact the instructor or TA.

### 3.4 Setting the Python Interpreter in VS Code

After activating the environment, tell VS Code to use the same Python:

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).
2. Type `Python: Select Interpreter`.
3. Select the interpreter at:
   ```
   /opt/conda_envs/course_env/bin/python
   ```
4. This only needs to be done once — VS Code remembers the selection.

### 3.5 Important Notes

- **Do not install packages yourself.** The environment is shared and read-only. If you believe a package is missing, contact the instructor or TA.
- **Always activate before running scripts.** Without activation, Python will use the system Python which does not have the course packages.
- **Slurm jobs also need activation.** Your job submission scripts must include the activation command (see Section 5).

---

## 4. Conda Environments — Background

> **Note:** This section is for your understanding. You do not need to create or manage conda environments for this course — the shared environment is already set up.

### 4.1 What is Conda?

Conda is a package and environment manager. It lets you create isolated Python environments — each with its own set of installed packages and Python version — so that different projects do not interfere with each other. Miniconda is a minimal installer for conda that does not include unnecessary extras.

### 4.2 Installing Miniconda (On Your Own Machine)

These instructions are for your local machine if you wish to set up Python outside the course.

1. Download the installer from [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). Choose the version for your operating system.
2. Run the installer:

- **Linux/macOS:**

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts and accept the defaults. When asked whether to initialize conda, type `yes`.

- **Windows:** Run the `.exe` installer and follow the prompts.

3. Close and reopen your terminal. You should see `(base)` at the beginning of your prompt, indicating conda is active.

4. Verify the installation:

```bash
conda --version
```

### 4.3 Creating and Managing Environments

To create a new environment with a specific Python version:

```bash
conda create -n my_env python=3.10 -y
```

Activate it:

```bash
conda activate my_env
```

Install packages:

```bash
pip install torch torchvision numpy matplotlib
```

Deactivate (return to `base`):

```bash
conda deactivate
```

Remove an environment entirely:

```bash
conda env remove -n my_env
```

---

## 5. Submitting GPU Jobs with Slurm

The GPU server is shared among all students. To ensure fair access, all GPU-intensive work (model training) must be submitted through **Slurm** — a job scheduling system. You do not run training scripts directly; you submit them to a queue, and Slurm assigns your job a GPU when one is available.

### 5.1 How It Works

1. You write a short submission script that says what to run and how long it needs.
2. You submit it with `sbatch`.
3. Slurm queues your job and runs it when a GPU is free.
4. Output and errors are written to log files.
5. When the job finishes, the GPU is released for the next student.

### 5.2 Job Submission Script

Each exercise includes a `submit_slurm.sh` file. A typical one looks like:

```bash
#!/bin/bash
#SBATCH --job-name=ex05
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=02:00:00

source /opt/miniconda3/bin/activate /opt/conda_envs/course_env

python src/train.py --config config.yaml
```

The `#SBATCH` lines configure the job:
- `--job-name`: a label for your job (visible in the queue)
- `--output` / `--error`: where stdout and stderr are saved
- `--time`: maximum wall time (HH:MM:SS) — the job is killed if it exceeds this

**Important:** The `source` line activates the course environment inside the job. Without it, Python won't find the course packages.

### 5.3 Submitting a Job

```bash
sbatch submit_slurm.sh
```

Slurm replies with a job ID:

```
Submitted batch job 42
```

### 5.4 Monitoring Your Job

```bash
# Check if your job is running or queued
squeue -u $USER

# View job output while it runs
tail -f outputs/logs/slurm_42.out

# View errors
cat outputs/logs/slurm_42.err
```

Job states:
- **PD** (Pending) — waiting for a GPU
- **R** (Running) — executing on a GPU
- **CG** (Completing) — finishing up
- **CD** (Completed) — done

### 5.5 Cancelling a Job

```bash
scancel <job_id>
```

### 5.6 Rules

- **One job at a time.** Each student can have one running job. Additional submissions will queue until the current one finishes.
- **Maximum 5 hours.** Jobs exceeding this limit are killed automatically.
- **Do not run training directly.** Running `python src/train.py` in your terminal (without Slurm) uses the GPU without scheduling, preventing other students from accessing it. Always use `sbatch`.
- **Quick tests are fine.** Short, non-GPU scripts (e.g. `python src/dataset.py` to verify your dataset loads) can run directly in the terminal.

---

## Summary

After completing this setup, your workflow for each exercise will be:

1. Open VS Code.
2. Connect to the GPU server via Remote-SSH.
3. Open your working folder on the server.
4. Activate the shared environment: `activate_course`
5. Write and run code.
6. Submit GPU training jobs: `sbatch submit_slurm.sh`
7. Monitor with: `squeue -u $USER`

You are now ready to begin Exercise 01.
