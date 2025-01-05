import numpy as np
import os
import pandas as pd
import re


test_df = pd.read_csv('test.csv')
repo_df = pd.read_csv('repo_metrics.csv')

repo_sorted = pd.merge(test_df[['project_b']], repo_df.rename(columns={'github_url': 'project_b'}), on='project_b',how='left')

repo_sorted.to_csv('repo_sorted.csv', index=False)
print("success")