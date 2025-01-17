# Predictive Challenge
An AI method to analyze different GitHub repositories and determine a relationship score between two GitHub repositories.

# Execution
install dependencies
```bash
pip install -r requirements.txt
```

# Current thought
Since using SVM and linear Regression get poorer result. It is obvious that it is not a linear problem.
Try using tree based model like Xgboost.



# Reference 
### paper
https://arxiv.org/pdf/2107.05112

https://openreview.net/pdf?id=vyRAYoxUuA

### Challenge link
https://huggingface.co/spaces/DeepFunding/PredictiveFundingChallengeforOpenSourceDependencies

### Example code
https://github.com/opensource-observer/insights/blob/main/community/dependency_graph_funding/Example_WeightedGraph.ipynb
