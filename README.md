# ACM Research Coding Challenge (Fall 2020)

## No Collaboration Policy

**You may not collaborate with anyone on this challenge.** You _are_ allowed to use Internet documentation. If you _do_ use existing code (either from Github, Stack Overflow, or other sources), **please cite your sources in the README**.

## Submission Procedure

Please follow the below instructions on how to submit your answers.

1. Create a **public** fork of this repo and name it `ACM-Research-Coding-Challenge`. To fork this repo, click the button on the top right and click the "Fork" button.
2. Clone the fork of the repo to your computer using . `git clone [the URL of your clone]`. You may need to install Git for this (Google it).
3. Complete the Challenge based on the instructions below.
4. Email the link of your repo to research@acmutd.co with the same email you used to submit your application. Be sure to include your name in the email.

## Question One

![Image of Cluster Plot](ClusterPlot.png)
<br/>
Given the following dataset in `ClusterPlot.csv`, determine the number of clusters by using any clustering algorithm. **You're allowed to use any Python library you want to implement this**, just document which ones you used in this README file. Try to complete this as soon as possible.

Regardless if you can or cannot answer the question, provide a short explanation of how you got your solution or how you think it can be solved in your README.md file.

## Answer

I decided to use one of scikit-learn's data clustering algorithms. The one I chose was DBSCAN (Density-based spatial clustering of applications with noise). As indicated by the name, it determines clusters by their density. My code reformats the .csv file and uses the DBSCAN algorithm as shown on scikit-learn's website. It determines that there are 3 clusters.

I also used sklearn's demo code to generate a plot visualzing the clusters in order to ensure that the parameters used were reasonable.

![Image of Clusters](ClusterIdentified.png)

Libraries:
- sklearn
- numpy
- matplotlib (for the visual reference; not in final code)

Referenced code from:
- [scikit-learn's demo](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py)
