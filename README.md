This repo contains part of the code for my Insight Data Science project. 

The goal of my project was to model healthcare expenses, and compare the effects of the cost sharing rules for different insurance plans on out-of-pocket expenses.

The implemented model is a random forest of regression trees, predicting billed expenses for the plan year.

The predictions are piped through the cost sharing rules for each of two plans being compared, and the output is a distribution of out-of-pocket expenses for each plan.

I wrote a string parser to extract the cost sharing rules for each plan.

To take a look at the project, see healthsavr.com , and healthsavr.com/slides for my demo.


