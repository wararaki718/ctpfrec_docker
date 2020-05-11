import numpy as np, pandas as pd
from ctpfrec import CTPF

## Generating a fake dataset
nusers = 10**2
nitems = 10**2
nwords = 5 * 10**2
nobs   = 10**4
nobs_bag_of_words = 10**4

np.random.seed(1)
counts_df = pd.DataFrame({
	'UserId' : np.random.randint(nusers, size=nobs),
	'ItemId' : np.random.randint(nitems, size=nobs),
	'Count'  : (np.random.gamma(1, 1, size=nobs) + 1).astype('int32')
	})
counts_df = counts_df.loc[~counts_df[['UserId', 'ItemId']].duplicated()].reset_index(drop=True)

words_df = pd.DataFrame({
	'ItemId' : np.random.randint(nitems, size=nobs_bag_of_words),
	'WordId' : np.random.randint(nwords, size=nobs_bag_of_words),
	'Count'  : (np.random.gamma(1, 1, size=nobs_bag_of_words) + 1).astype('int32')
	})
words_df = words_df.loc[~words_df[['ItemId', 'WordId']].duplicated()].reset_index(drop=True)

## Fitting the model
recommender = CTPF(k = 15, reindex=True)
recommender.fit(counts_df=counts_df, words_df=words_df)

## Making predictions
recommender.topN(user=10, n=10, exclude_seen=True)
recommender.topN(user=10, n=10, exclude_seen=False, items_pool=np.array([1,2,3,4]))
recommender.predict(user=10, item=11)
recommender.predict(user=[10,10,10], item=[1,2,3])
recommender.predict(user=[10,11,12], item=[4,5,6])

## Evaluating Poisson log-likelihood
recommender.eval_llk(counts_df, full_llk=True)

## Adding new items without refitting
nitems_new = 10
nobs_bow_new = 2 * 10**3
np.random.seed(5)
words_df_new = pd.DataFrame({
	'ItemId' : np.random.uniform(low=nitems, high=nitems+nitems_new, size=nobs_bow_new),
	'WordId' : np.random.randint(nwords, size=nobs_bow_new),
	'Count' : np.random.gamma(1, 1, size=nobs_bow_new).astype('int32')
	})
words_df_new = words_df_new.loc[words_df_new.Count > 0]

recommender.add_items(words_df_new)