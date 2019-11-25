# cs538-project

This repo contains the code for the CSE 538 Natural Language Processing course project.

Team members:

- Nikita Soni
- Nishant Shankar
- Siddhant Rele

## Setup

 Since this repo uses submodules, clone the repository in the following manner.

`git clone --recurse-submodules https://github.com/echodarkstar/cs538-project.git`

 This repo requires Pytorch. You can install it via Anaconda by following [these instructions](https://pytorch.org/). Other dependencies can be installed by `pip install -r requirements.txt` inside a conda enviornment.

If you're modifying the submodule, the changes can be pushed to the submodule repo. Add the untracked submodule to the main repo and then push changes.

[Submodule reference](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

### Notes

- After installing both the dependencies of universal-triggers and transfer-conv-ai, the pytorch-transformers versions get messed up. Just do `pip install --upgrade --no-deps  pytorch-transformers==1.2.0` to make sure transfer-conv-ai works properly (this doesn't mess up universal-triggers). You might also have to download the en model via spacy (`python -m spacy download en`).

- A messy aspect of using submodules is that some imports are buggy. Adding a . in front helps in resolving the issue though, but this is a very hacky and temporary fix. Be aware of how imports are happening (or move away from submodules...)