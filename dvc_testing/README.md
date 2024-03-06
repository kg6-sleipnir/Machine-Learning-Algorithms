# DVC Testing
This directory contains tests using [DVC](https://dvc.org/) for managing ML models.

The general purpose of DVC is to provide interfaces for performing reproducible
experiments using files containing metadata about models, datasets, and results
that can be version controlled using a tool like git.

DVC also allows for specifying pipelines that perform sequences of actions such
as `preprocess->train->validate`. This feature will also keep track of already
performed steps in a manner similar to Makefiles.

There is also a feature where different experiments can be run. After running an
experiment, the results can be saved to a file to record using a version control
system. These results can also be plotted in `.html` files that can be opened in
a browser.