# Differential Expression Viewer
Visualization of differential expression results

## Background

A useful approach in the analysis of biological systems is to compare the relative
level of gene expression between groups of biological samples.
This process is generally called differential expression analysis, and produces
a typical set of results for each gene included in the analysis:

- Fold Change (log2): The log-transformed ratio between the expression levels observed in different groups
- p-value: The inferred probability that the observed values are the result of a random process
- Mean Expression: The average expression across all samples

## Visualizations

To inspect these results, researchers tend to make:

- Volcano plot: Comparing the fold change and p-value for each gene
- MA plot: Comparing the fold change and mean expression for each gene

Between these two plots, it is generally possible to identify genes which are
expressed at significantly different levels between two groups, while also
distinguishing between genes that are highly and lowly expressed overall.

## Implementation

To provide a quick visualization of this type of data, this repository contains
an interactive app built using [marimo](https://marimo.io).

## Development

Set up your development environment:

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Launch the app in editable notebook format:

```
marimo edit diffexp.marimo.py
```