# Exercise 01: Introduction to Git and GitHub

## Overview

This exercise introduces you to Git and GitHub — the version control tools you will use throughout this course to manage your code, track changes, and submit your work. By the end of this exercise, you will be comfortable with the basic Git workflow that every subsequent exercise depends on.

**Estimated time:** 60–90 minutes

---

## 1. Prerequisites

### What is Git?

Git is a tool that tracks changes to files over time. Think of it as a detailed "undo history" for your entire project — you can see what changed, when, and by whom. It works locally on your computer.

### What is GitHub?

GitHub is a website that hosts Git repositories (projects) online. It lets you back up your work remotely, share it, and collaborate with others. Git is the tool; GitHub is the platform.

### 1.1 Install Git

- **Windows:** Download and install from [git-scm.com](https://git-scm.com/downloads). Use the default options during installation.
- **macOS:** Open Terminal and run `git --version`. If Git is not installed, you will be prompted to install it.
- **Linux:** Run `sudo apt install git` (Ubuntu/Debian) or `sudo dnf install git` (Fedora).

### 1.2 Create a GitHub Account

If you do not already have one, create a free account at [github.com](https://github.com).

### 1.3 Configure Git

Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run the following two commands, replacing the placeholder values with your actual name and email:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

This tells Git who you are. Every change you make will be tagged with this identity.

### Task 1

Run the following command and confirm it displays your name and email correctly:

```bash
git config --list
```

---

## 2. Fork and Clone the Course Repository

### What is a Fork?

A fork is your personal copy of someone else's repository on GitHub. When you fork the course repository, you get your own version of it under your GitHub account. Changes you make to your fork do not affect the original.

### What is Cloning?

Cloning downloads a repository from GitHub to your local machine so you can work on it. The cloned folder on your computer is linked to the remote repository on GitHub.

### Task 2

1. Go to the course repository start page on GitHub ([here](https://github.com/smAIL-WS/AI2-exercise-tasks)).
2. Click the **Fork** button in the top-right corner. This creates a copy under your own GitHub account.
3. On **your forked repository** page, click the green **Code** button and copy the HTTPS URL.
4. Open a terminal, it should open in your home directory.
5. Clone your forked repository into this folder:

```bash
git clone <paste-your-fork-url-here>
```

6. Navigate into the cloned folder and into the exercise 01 subfolder:

```bash
cd <repository-name>/exercise_01
```

You now have a local copy of the course repository on your machine inside your home directory, linked to your fork on GitHub.

---

## 3. The Core Workflow: Modify → Stage → Commit → Push

This is the workflow you will repeat in every exercise. Understand these four steps and you understand 90% of daily Git usage.

| Step | Command | What it does |
|------|---------|--------------|
| **Modify** | (edit files normally) | Make changes to files using any text editor |
| **Stage** | `git add <file>` | Mark specific changes to be included in the next snapshot |
| **Commit** | `git commit -m "message"` | Save a snapshot of all staged changes with a description |
| **Push** | `git push` | Upload your local commits to GitHub |

**Why is staging separate from committing?** Staging lets you choose exactly which changes go into a commit. You might edit five files but only want to commit two of them right now — staging gives you that control.

### Task 3

1. Inside your cloned repository, navigate to the `exercise_01/` folder.
2. Create a new file called `about_me.txt` using any text editor. Write one or two lines about yourself (name, program of study, etc.).
3. Stage the file:

```bash
git add exercise_01/about_me.txt
```

4. Commit with a descriptive message:

```bash
git commit -m "Add about_me.txt with personal introduction"
```

5. Push the commit to your fork on GitHub:

```bash
git push
```

6. Go to your fork on GitHub in a browser and verify that `exercise_01/about_me.txt` is now visible there.

---

## 4. Checking Status and History

Two commands you will use constantly to understand what is happening in your repository.

**`git status`** shows the current state of your working directory: which files have been modified, which are staged, and which are untracked (new files Git does not know about yet).

**`git log`** shows the history of commits — who made each change, when, and the commit message they wrote. Press `q` to exit the log view.

### Task 4

1. Open `exercise_01/about_me.txt` and add a new line (e.g. one hobby or interest). Save the file.
2. Run `git status`. Observe that the file appears under "Changes not staged for commit."
3. Stage and commit this change:

```bash
git add exercise_01/about_me.txt
git commit -m "Add hobby to about_me.txt"
```

4. Make another small edit to the same file (e.g. add your favorite programming language, even if you do not have one yet). Stage and commit again with a different message.
5. Run `git log` and observe your three commits (the initial one from Task 3, plus the two you just made). Each has a unique hash, your name, a timestamp, and your message.
6. Push all commits:

```bash
git push
```

---

## 5. Branches

### What is a Branch?

A branch is an independent line of work. By default, your repository has one branch called `main`. When you create a new branch, you can make changes there without affecting `main`. Once you are satisfied, you merge the branch back into `main`.

**Why use branches?** They let you experiment or work on a specific feature without risking the stable version of your code. In later exercises, you might use a branch to try a different model architecture while keeping your working version safe on `main`.

### Task 5

1. Create a new branch called `experiment` and switch to it:

```bash
git checkout -b experiment
```

2. Create a new file called `exercise_01/experiment.txt` and write anything in it (e.g. "This is a test file on a branch.").
3. Stage and commit:

```bash
git add exercise_01/experiment.txt
git commit -m "Add experiment.txt on experiment branch"
```

4. Switch back to the `main` branch:

```bash
git checkout main
```

5. Check the contents of the `exercise_01/` folder — notice that `experiment.txt` is **not there**. It only exists on the `experiment` branch.

6. Merge the `experiment` branch into `main`:

```bash
git merge experiment
```

7. Check the folder again — `experiment.txt` is now on `main`.
8. Push the result:

```bash
git push
```

---

## 6. Pulling Changes

### What is `git pull`?

`git pull` downloads the latest changes from GitHub and merges them into your local branch. It is the opposite of `git push` — push sends your changes up, pull brings changes down.

**When do you need it?**

- When the instructor updates the course repository and you need to get the latest exercises or fixes.
- When you edited files on GitHub's web interface and need those changes on the server.
- When you work from multiple machines (e.g. your laptop and the GPU server) and need to sync between them.

### How it works

```bash
git pull
```

This is shorthand for two steps combined: `git fetch` (download the changes) followed by `git merge` (integrate them into your current branch).

### Pulling from the original course repository

Your fork is a snapshot of the course repository at the time you forked it. If the instructor adds or updates exercises later, your fork does not get those changes automatically. To pull updates from the original repository:

1. Add the original repository as a remote (one-time setup):

```bash
git remote add upstream <original-repository-url>
```

2. Fetch and merge the latest changes:

```bash
git fetch upstream
git merge upstream/main
```

3. Push the merged changes to your fork:

```bash
git push
```

### When `git pull` fails: merge conflicts

If you modified a file locally and the same file was changed on GitHub, Git cannot merge them automatically. This is called a **merge conflict**. Git will mark the conflicting sections in the file:

```
<<<<<<< HEAD
Your local changes
=======
Changes from GitHub
>>>>>>> origin/main
```

To resolve it: open the file, choose which version to keep (or combine them), remove the conflict markers, then stage and commit.

For this course, conflicts are rare — you typically work alone on your fork. But if they happen, ask the instructor or TA for help.

### Task 6

1. Go to your fork on GitHub in a browser.
2. Open `exercise_01/about_me.txt` and click the pencil icon to edit it.
3. Add a new line (e.g. "Edited from GitHub") and commit directly on GitHub.
4. Back in your terminal on the server, run:

```bash
git pull
```

5. Open `about_me.txt` locally and verify the line you added on GitHub is now there.
6. Run `git log` — you should see the commit you made on GitHub in your local history.

---

## 7. Using .gitignore

### Why .gitignore Matters

In future exercises, your project folders will contain files that should **not** be tracked by Git: large dataset files, model checkpoints, temporary Python cache folders, and environment-specific files. Committing these wastes storage and clutters your repository. A `.gitignore` file tells Git which files and folders to ignore.

### Task 7

1. In the root of your repository, create a file named `.gitignore` (note the dot at the beginning).
2. Add the following lines to it:

```
# Python cache
__pycache__/
*.pyc

# Datasets (will be symlinked in later exercises)
data/

# Model checkpoints
*.pth
*.pt
saved/

# Jupyter notebook checkpoints
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db
```

3. Now create a test file to verify it works. Create a file called `test_ignore.pyc` in the root of your repository.
4. Run `git status`. The `.gitignore` file should appear as untracked, but `test_ignore.pyc` should **not** appear — Git is ignoring it.
5. Stage, commit, and push the `.gitignore` file:

```bash
git add .gitignore
git commit -m "Add .gitignore for Python and ML artifacts"
git push
```

6. Delete `test_ignore.pyc` — it was only for testing.

---

## 8. Quick Reference

A summary of every command used in this exercise. Refer back to this in future exercises.

| Command | What it does |
|---------|--------------|
| `git config --global user.name "Name"` | Set your name for commits |
| `git config --global user.email "email"` | Set your email for commits |
| `git clone <url>` | Download a remote repository to your machine |
| `git status` | Show the current state of your files |
| `git add <file>` | Stage a file for the next commit |
| `git add .` | Stage all changed files |
| `git commit -m "message"` | Save a snapshot with a description |
| `git push` | Upload local commits to GitHub |
| `git log` | View commit history |
| `git checkout -b <branch>` | Create and switch to a new branch |
| `git checkout <branch>` | Switch to an existing branch |
| `git merge <branch>` | Merge a branch into the current branch |
| `git pull` | Download and merge remote changes into your local branch |
| `git remote add upstream <url>` | Link the original course repo for pulling updates |
| `git fetch upstream` | Download changes from the original repo (without merging) |

---

## Summary

You now know the Git workflow that you will use for every remaining exercise in this course:

1. **Clone/pull** the latest version of your repository.
2. **Create a branch** if experimenting (optional but recommended).
3. **Modify** files — write code, edit configs.
4. **Stage** your changes with `git add`.
5. **Commit** with a clear message using `git commit -m "..."`.
6. **Push** to GitHub with `git push`.

From Exercise 02 onward, your work will follow this exact cycle. The better this becomes muscle memory now, the less friction you will face later.
