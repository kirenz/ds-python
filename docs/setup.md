## Python Setup

First you will need to have Python installed (choose the lates version):

- [Download Python](https://www.python.org/downloads/)

We use [`pyenv`](https://github.com/pyenv/pyenv) to manage the version of Python we use for each project and  pair [`pyenv`](https://github.com/pyenv/pyenv-virtualenv) with `pyenv-virtualenv` in order to use pyenv for managing both the python version and our virtual environments. 

When we start a new project, we first create a virtual environment with a specific version of Python:

```Python
# Create a virtual environment called "my-new-project"
# using Python 3.8.8
pyenv virtualenv 3.8.8 my-new-project
# Activate the virtual environment
pyenv activate my-new-project
```