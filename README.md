# metalog
Sergey Kim, Reidar Brumer Bratvold


## Metalog Distribution

The metalog distributions constitute a new system of continuous univariate probability distributions designed for flexibility, simplicity, and ease/speed of use in practice. The system is comprised of unbounded, semi-bounded, and bounded distributions, each of which offers nearly unlimited shape flexibility compared to Pearson, Johnson, and other traditional systems of distributions.

The following [paper](http://www.metalogdistributions.com/images/TheMetalogDistributions.pdf) and [website](http://www.metalogdistributions.com/home.html) provide a full background of the metalog distribution.


## Using the Package

This Python package was transfered from [RMetalog](https://github.com/isaacfab/RMetalog) package and therefore shares the same R-based structure.

The [data](https://www.sciencebase.gov/catalog/item/5b45380fe4b060350a140b7b) used for demonstration are body length of salmon and was collected in 2008-2010:

```
import numpy as np
import pandas as pd

salmon = np.array(pd.read_csv("salmon.csv")['Length'])
```
