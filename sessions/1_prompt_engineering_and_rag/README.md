# AI in the news industry - prompting and RAG

In this two-part series, we will learn basic generative AI basics like prompt tuning and retrieval
augmented generation (RAG).

## Setup

Before you can start with this exercise, you need to setup a Kaggle account and a Google AI studio account. If you don't have those accounts yet, please complete the steps below.

**Create Google AI Studio account:**

We use LLM APIs from [Google AI Studio](https://aistudio.google.com/). Google offers limited LLM quotas for free, all you need is a Google account. Please follow these steps:
- Open [Google AI Studio](https://aistudio.google.com/welcome) in your web browser
- Click on ``Sign in to Google AI Studio``. If you are already logged in, click on ``Go to Google AI Studio`` instead.
- Once logged in with a Google account, click on ``Get API key``, then ``Create API key``.
- Remember your API key, you'll need it for running the hands-on exercises

**Create Kaggle account and start Jupyter notebook session:**
- Before starting, you need to clone the workshop repository to your local machine:
```
git clone https://github.com/thomsonreuters/labs_AMLD25_Workshop.git
```
- Go to the [Kaggle](https://www.kaggle.com/) web site
- If you don't have a Kaggle account yet, click on ``Register`` → ``Register with Google`` → follow the instructions.
- If you already have a Kaggle account, click on ``Sign In`` instead, and follow the instructions.
- You need a verified Kaggle account to enable GPU support.
- To verify your Kaggle account, go to your Kaggle profile (top right profile icon -> click on ``Your Profile``).
Next, click on ``Settings`` -> ``Phone verification``. Follow the steps using a valid phone number.
- Now you're ready to start up a Jupyter notebook. Click on `Code` → ``+ New Notebook``
- Click on ``File`` tab → ``Import Notebook``
- Upload the appropriate Jupyter notebook (e.g. `sessions/1_prompt_engineering_and_rag/part1_news_qa/news_qa.ipynb`)

## Run instructions

- Start Jupyter kernel on Kaggle
- Enable GPU support: In your active kernel, click on ``Settings`` -> ``Accelerator`` -> select ``GPU P100``.
- Ensure you have your Google AI Studio API key ready

## Notebooks overview
There are two hands on exercises you can choose from. You may not have
time to complete both, so choose which one you want to start with:

- Part 1 -`part1_news_qa/news_qa.ipynb`: In this exercise, you will create a Q&A system for Reuters news articles.
Along the way, you leverage different RAG techniques, including semantic search and TF-IDF, which you implement from scratch.
- Part 2 -`part2_quiz_generation/quiz_generation.ipynb`: Here you implement a quiz generator for news articles.
This exercise focuses on prompt engineering. Along the way, you'll learn about a library called LangGraph.