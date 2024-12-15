# ADS-Sentiment-Analysis-and-Performance-Experiments-in-Snowflake

**Author: Carl Bruun (carbr@itu.dk)**

This project includes all SQL used through Snowflake in accordance with project-1 of Advanced Data Systems at ITU (2024, MSc., ComSci).

## Files:

- **NaiveBayesSQL.sql**
Includes all SQL for the first part of the project, e.g. performing sentiment analysis using Bayes theorem over 700k [yelp_reviews](https://huggingface.co/datasets/Yelp/yelp_review_full) from HuggingFace. That is, predict the sentiment/score of a review given the review text.
This SQL uses a Python UDF (user-defined-function) to clean and split review text, as found in **clean.py**, including test case. 

- **NaiveBayesUDTF.sql**
Includes all SQL to complete second part of the project, e.g. performing sentiment analysis now using UDTF (user-defined-table-functions). That is, this SQL includes a vectorized Python UDTF as found in **sentiment_vectorized.py**. 

- **TPCH.sql**
Includes all SQL for the third and final part of the project, e.g. performing TPC-H benchmarks using varying warehouse configurations and scaling factors against snowflake with TPC-H queries: 1, 5, and 18. 


## Notes:
All Naive Bayes implementations are based on Stanford Lecture Slides [(click here)](https://web.stanford.edu/class/cs124/lec/naivebayes2021.pdf)
