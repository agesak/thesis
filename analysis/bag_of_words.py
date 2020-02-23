import pandas as pd
import glob
import os
import numpy as np
from mcod_prep.utils.mcause_io import get_mcause_data
from thesis_utils.directories import get_limited_use_directory
from thesis_data_prep.launch_mcod_mapping import MCauseLauncher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

BLOCK_RERUN = {'block_rerun': False, 'force_rerun': True}

# not limited use data
unlimited = get_mcause_data(
    phase='format_map', source=['COL_DANE', 'ZAF_STATSSA'], sub_dirs=f"{int_cause}/thesis",
    data_type_id=9, assert_all_available=True,
    verbose=True, **BLOCK_RERUN)

dfs = []
for source in MCauseLauncher.limited_sources:
    print(source)
    limited_dir = get_limited_use_directory(source, int_cause)
    csvfiles = glob.glob(os.path.join(limited_dir, '*.csv'))
    for file in csvfiles:
        print(file)
        df = pd.read_csv(file)
        dfs.append(df)
limited = pd.concat(dfs, ignore_index=True, sort=True)

df = pd.concat([unlimited, limited], sort=True, ignore_index=True)


test_df = df[["cause_id", "cause_info", "x59"]]

# subset to 300 rows so can run predict on rest
# this isnt a real training set (not rules followed here)
train = test_df.query("x59==0").head(300)

# do i need to specify min_df?
cv = CountVectorizer(lowercase=False)
tf = cv.fit_transform(train["cause_info"])
# naive bayes
clf = MultinomialNB().fit(tf, train["cause_id"])
test = test_df.query("x59==0").tail(200)
# is there a way to see how many elements of this arent in the vocabulary?
new_counts = cv.transform(test["cause_info"])
predicted = clf.predict(new_counts)
test["predicted"] = predicted
np.mean(test.predicted == test.cause_id)