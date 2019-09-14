# metalog
Sergey Kim, Reidar Brumer Bratvold


## Metalog Distribution

The metalog distributions constitute a new system of continuous univariate probability distributions designed for flexibility, simplicity, and ease/speed of use in practice. The system is comprised of unbounded, semi-bounded, and bounded distributions, each of which offers nearly unlimited shape flexibility compared to Pearson, Johnson, and other traditional systems of distributions.

The following [paper](http://www.metalogdistributions.com/images/TheMetalogDistributions.pdf) and [website](http://www.metalogdistributions.com/home.html) provide a full background of the metalog distribution.


## Using the Package

This Python package was transfered from [RMetalog](https://github.com/isaacfab/RMetalog) package and therefore shares the same R-based structure.

The [data](https://www.sciencebase.gov/catalog/item/5b45380fe4b060350a140b7b) used for demonstration are body length of salmon and were collected in 2008-2010:

```
import numpy as np
import pandas as pd

salmon = chinook = pd.read_csv("Chinook and forage fish lengths.csv")

# Filtered data for eelgrass vegetation and chinook salmon
chinook = chinook[(chinook['Vegetation'] == 'Eelgrass') & (chinook['Species'] == 'Chinook_salmon')]
```

To import package with metalog distribution run the code:

```
from metalog import metalog
```

To fit the data to metalog distribution one should use function ```metalog.fit()```. It has the following arguments:

- ```x```: data.

- ```bounds```: bounds of metalog distribution. Depending on ```boundedness``` argument can take zero, one or two values.

- ```boundedness```: boundedness of metalog distribution. Can take values ```'u'``` for unbounded, ```'sl'``` for semi-bounded lower, ```'su'``` for semi-bounded upper and ```'b'``` for bounded on both sides.

- ```term_limit```: maximum number of terms to specify the metalog distribution. Can take values from 3 to 30.

- ```term_lower_bound```: the lowest number of terms to specify the metalog distribution. Must be greater or equal to 2 and less than ```term_limit```

- ```step_len```: size of steps to summarize the distribution.

- ```probs```: probabilities corresponding to data.

- ```fit_method```: fit method ```'OLS'```, ```'LP'``` or ```'any'```.

- ```save_data```: if ```True``` then data will be saved for future update.

