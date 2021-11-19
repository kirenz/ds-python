# Programming toolkit

This section contains an overview about the programming toolkit you will need for our course. Please read the instructions and complete the tasks listed in the yellow *To do* boxes.

## Python

Python is an object-oriented language (an object is an entity that contains data along with associated metadata and/or functionality). One thing that distinguishes Python from other programming languages is that it is interpreted rather than compiled. This means that it is executed line by line which is particular useful for data analysis, as well as the creation of interactive, executable documents like Jupyter Notebooks.

On top of this, there is a broad ecosystem of third-party tools and libraries that offer more specialized data science functionality.

## Jupyter Notebook

:::{note}
Jupyter Notebook is a web-based interactive computational environment for creating documents that contain code and text
:::

[Jupyter Notebook](https://jupyter.org/) is an open-source web application that allows you to create and share documents that contain code, equations, visualizations and narrative text:

- A notebook is basically a list of cells
- Cells contain either
  - explanatory text or
  - executable code and its
  - output

## Colab

:::{note}
Colab is a free Jupyter notebook environment that requires no setup, and runs entirely on the Cloud.
:::

Colaboratory, or “Colab” for short, is a free to use product from Google Research. Colab allows anybody to write and execute python code through the browser, and is especially well suited to perform data analysis and machine learning.

Watch this video to get a first impression of Colab:

<iframe width="560" height="315" src="https://www.youtube.com/embed/inN8seMm7UI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Let`s start your first Colab notebook to get an overview about some basic features:

```{admonition} To do
:class: tip
- [Colab basic features overview](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
```

## Markdown

:::{note}
Markdown is a simple way to format text that looks great on any device. 
:::

Markdown is one of the world’s most popular markup languages used in data science. Jupyter Notebooks use Markdown to provide an unified authoring framework for data science, combining code, its results, and commentary in Markdown.  

According to {cite:t}`wickham2016`, Markdown files are designed to be used in three ways:

1. For communicating to decision makers, who want to focus on the conclusions, not the code behind the analysis.

2. For collaborating with other data scientists, who are interested in both your conclusions, and how you reached them (i.e. the code).

3. As an environment in which to do data science, as a modern day lab notebook where you can capture not only what you did, but also what you were thinking.

Review this sites to learn more about Markdown:


```{admonition} To do
:class: tip
- [Interactive Colab Markdown guide](https://colab.research.google.com/notebooks/markdown_guide.ipynb)

- [Interactive 10 minute Markdown tutorial](https://commonmark.org/help/)
```

## Libraries

:::{note}
Python Libraries are a set of useful functions that eliminate the need for writing codes from scratch.
:::

A Python library is a reusable chunk of code that you can import in your own projects so you don't have to write all the code by yourself. There are around 140000 available Python projects and one way to discover and install them is to use the [Python Package Index (PyPI)](https://pypi.org/). Another way to install Python libraries is to use the open source data science platform Anaconda, which will be covered below.

Here a list of some of the libraries we will use frequently:

- [pandas](https://pandas.pydata.org/) is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.

- [NumPy](https://numpy.org/) offers tools for scientific computing like mathematical functions and random number generators.

- [SciPy](https://scipy.org/) contains algorithms for scientific computing.

- [matplotlib](https://matplotlib.org/) is a library for creating data visualizations.

- [Seaborn](https://seaborn.pydata.org/) provides a high-level interface for drawing attractive and informative statistical graphics.

- [plotly](https://plotly.com/python/) is a graphing library to make interactive, publication-quality graphs.

- [statsmodels](https://www.statsmodels.org/stable/index.html) includes statistical models, hypothesis tests, and data exploration.

- [scikit-learn](https://scikit-learn.org/stable/) provides a toolkit for applying common machine learning algorithms to data.

- [TensorFlow](https://www.tensorflow.org/) is s an end-to-end open source platform for machine learning.

Here are two curated lists with an extensive list of resources for practicing data science using Python, including not only libraries, but also links to tutorials, code snippets, blog posts and talks:

- [Awesome Data Science with Python](https://github.com/r0f1/datascience)
- [Awesome Python Data Science](https://github.com/krzjoa/awesome-python-data-science)

## Anaconda

:::{note}
Anaconda is a data science toolkit which already includes most of the libraries we need.
:::

The open-source [Anaconda](https://www.anaconda.com/products/individual) Individual Edition (Distribution) is on of the easiest ways to perform Python and R data science and machine learning since it already includes Python and the most important packages and libraries we need. In particular, it already contains Jupyter Notebook and other important data science modules.

 Furthermore, Anaconda's package manager `conda` makes it easy to manage multiple data environments that can be maintained and run separately without interference from each other (in so called virtual environments). `conda` analyses the current environment including everything currently installed, and, together with any version limitations specified (e.g. the user may wish to have TensorFlow version 2,0 or higher), works out how to install a compatible set of dependencies, and shows a warning if this cannot be done.


```{admonition} To do
:class: tip

Install Anaconda Individual Edition

- [Anaconda installation tutorial](https://kirenz.github.io/codelabs/codelabs/anaconda-install/#0)
```

Here an example of how to install the Python package seaborn using `conda`:

- On *Windows* open the Start menu and open an Anaconda Command Prompt. 
- On *macOS* or *Linux* open a terminal window.
- Activate the conda environment of your choice (e.g. the base environment)

```bash
conda activate base
```

- Install seaborn according to the [documentation](https://anaconda.org/anaconda/seaborn)

```bash
conda install -c anaconda seaborn
```

## Visual Studio Code 

:::{note}
Visual Studio Code is a code editor that can be used with a variety of programming languages including Python.
:::

Visual Studio Code (also called Code) is a powerful source code editor which runs on your desktop and is available for Windows, macOS and Linux. It comes with a rich ecosystem of extensions for Python and we use them to write our Pyton code.

```{admonition} To do
:class: tip

Install VS Code:

- [Install Code](https://code.visualstudio.com/)

Install Extensions:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint)
- [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv)

Learn how to use Jupyter Notebooks:

- [How to use Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
```