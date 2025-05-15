# AI in the news industry - prompting and RAG

In this two-part series, we will learn basic generative AI basics like prompt tuning and retrieval
augmented generation (RAG).

This part of the workshop is self-contained. Please disregard the instructions in the README file one folder above, and instead follow all instructions in this file here. In particular, for this workshop part, please use Google Colab, rather than Kaggle, as run environment.

## Setup

Before you can start with this exercise, you are encouraged to read this whole README file, plus you need to setup a Google AI studio account.

**Have a look at the repository**

- Before starting, we recommend you have a look this Git repository.
- You can browse through the repository on [GitHub](https://github.com/thomsonreuters/labs_AMLD25_Workshop), or you can download the repo to you laptop
- Here's how you can download the repo to your local machine:
```
git clone https://github.com/thomsonreuters/labs_AMLD25_Workshop.git
```

**Create Google AI Studio account:**

We use LLM APIs from [Google AI Studio](https://aistudio.google.com/). Google offers limited LLM quotas for free, all you need is a Google account. Please follow these steps:
- Open [Google AI Studio](https://aistudio.google.com/welcome) in your web browser
- Click on ``Sign in to Google AI Studio``. If you are already logged in, click on ``Go to Google AI Studio`` instead.
- Once logged in with a Google account, click on ``Get API key``, then ``Create API key``.
- Remember your API key, you'll need it for running the hands-on exercises

## Run exercises on Google Colab

We use Google Colab to run Jupyter Notebooks. The steps below indicate how you can run and experiment with the hands-on exercises:

- Go to the [Google Colab](https://colab.research.google.com/) website
- We recommend you log in with your Google account (``Sign In``)
- Enable GPU support: Pick menu ``Runtime``, then click ``Change runtime type`` -> choose any GPU hardware accelerator like ``T4 GPU``
- Import an exercise: Click on menu ``File``, ``Open notebook``, then ``GitHub``, enter our GitHub URL (https://github.com/thomsonreuters/labs_AMLD25_Workshop).
- Choose the appropriate Notebook. For a description of the two available exercises, read the next section.
- Ensure you have your Google AI Studio API key ready
- Now you're ready to execute the first notebook cell

## Notebooks overview
There are two hands on exercises you can choose from. You may not have
time to complete both, so choose which one you want to start with:

- Part 1 -`part1_news_qa/news_qa.ipynb`: In this exercise, you will create a Q&A system for Reuters news articles.
Along the way, you leverage different RAG techniques, including semantic search and TF-IDF, which you implement from scratch.
- Part 2 -`part2_quiz_generation/quiz_generation.ipynb`: Here you implement a quiz generator for news articles.
This exercise focuses on prompt engineering. Along the way, you'll learn about a library called [LangGraph](https://www.langchain.com/langgraph).

## Solutions

You can find the full solutions to both hands-on exercises in the `solutions` folder.
