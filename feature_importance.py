# Permutation based feature importance
# source: https://github.com/nestordemeure/permutationImportance/feature_importance.py

from tqdm import tqdm
from fastai.tabular import *

__all__ = ['feature_importance', 'plot_feature_importance']

def log_magnitude(x):
    "log that is still valid for negative inputs"
    return np.sign(x)*np.log1p(abs(x))

def feature_importance(learn:Learner, datasetType:DatasetType=DatasetType.Valid):
    "Takes a learned, a type of dataset and returns a dataframe with the importance of each features."
    pd.options.mode.chained_assignment = None
    # selects target dataset loader
    if datasetType == DatasetType.Valid: dl = learn.data.valid_dl
    elif datasetType == DatasetType.Train: dl = learn.data.train_dl
    elif datasetType == DatasetType.Test: dl = learn.data.test_dl
    else: raise Exception("This DatasetType cannot be used!")
    # computes the baseline loss
    data = learn.data.train_ds.x
    cat_names = data.cat_names
    cont_names = data.cont_names
    loss0 = np.array([learn.loss_func(learn.pred_batch(batch=(x,y.to("cpu"))), y.to("cpu")) for x,y in iter(dl)]).mean()
    #The above gives us our ground truth percentage for our validation set
    fi=dict()
    types=[cat_names, cont_names]
    with tqdm(total=len(data.col_names)) as pbar:
      for j, t in enumerate(types): # for all of cat_names and cont_names
        for i, c in enumerate(t):
          loss=[]
          for x,y in (iter(dl)): # for all values in validation set
            col=x[j][:,i] # select one column of tensors
            idx = torch.randperm(col.nelement()) # generate a random tensor
            x[j][:,i] = col.view(-1)[idx].view(col.size()) # replace the old tensor with a new one
            y=y.to('cpu')
            loss.append(learn.loss_func(learn.pred_batch(batch=(x,y)), y))
          pbar.update(1)
          fi[c]=np.array(loss).mean()-loss0
    d = sorted(fi.items(), key=lambda kv: kv[1], reverse=False)
    # builds output dataframe
    df = pd.DataFrame({'Variable': [l for l, v in d], 'Importance': [log_magnitude(v) for l, v in d]})
    df['Type'] = ''
    for x in range(len(df)):
      if df['Variable'].iloc[x] in cat_names:
        df['Type'].iloc[x] = 'categorical'
      if df['Variable'].iloc[x] in cont_names:
        df['Type'].iloc[x] = 'continuous'
    return df

def plot_feature_importance(learn:Learner, datasetType:DatasetType=DatasetType.Valid, title="Feature importance", **plot_kwargs):
    "Plots the feature importance for a given learner and dataset type (additional inputs will be passed to the Pandas plotting function)"
    importances = feature_importance(learn, datasetType)
    return importances.plot('Variable', 'Importance', kind='barh', title=title, legend=False, **plot_kwargs)

