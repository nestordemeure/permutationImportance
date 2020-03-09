# Permutation based feature importance

This code estimate the importances of the features of a fastai tabular learner model using the [permutation method](https://christophm.github.io/interpretable-ml-book/feature-importance.html).

While I made some modifications to it (adding functionalities and fixing bugs) most of the credits goes to [Miguel Mota Pinto](https://medium.com/@mp.music93/neural-networks-feature-importance-with-fastai-5c393cf65815) for the first prototype, [Zachary Mueller](https://forums.fast.ai/t/feature-importance-in-deep-learning/42026/21) for the improved version and [John Keefe](https://johnkeefe.net/detecting-feature-importance-in-fast-dot-ai-neural-networks) for the plotting function.

## Usage

```python
from feature_importance import *

# gets feature importances as a dataframe
importances = feature_importance(learn)

# note that you can, optionnally, specify the dataset on which you want to compute the importances (validation dataset by default)
importances_train = feature_importance(learn, DatasetType.Train)

# plots feature importances directly
plot_feature_importance(learn)
```

## Todo

- the code severely needs refactoring to improve readability
- no need to sort features when computing feature importance (better done only if the user needs it for plotting purposes)
- could output a class that inherit from a dataframe but with an overloaded plotting function

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*
