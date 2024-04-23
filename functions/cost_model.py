"""
Author: Benedikt Goodman, Division for National Accounts, Statistics Norway
Email: bgo@ssb.no
Created: 22/04/2024
"""

import pandas as pd
import numpy as np


def make_df_cost(employees, calls, q_length, a_length, tokens_per_word):
    """
    Generates a DataFrame containing all possible combinations of specified parameters related
    to token usage in a hypothetical scenario where employees handle calls. The DataFrame estimates
    the millions of tokens sent and received based on the number of calls, length of questions,
    length of answers, and the tokens per word.

    Parameters
    ----------
    employees : numpy.array
        An array of integers, each representing the number of employees handling the calls.
    calls : numpy.array
        An array of integers, each representing the number of calls handled.
    q_length : numpy.array
        An array of integers where each integer represents the average number of words in a question.
    a_length : numpy.array
        An array of integers where each integer represents the average number of words in an answer.
    tokens_per_word : numpy.array
        An array of floats or integers representing the number of tokens generated per word.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns representing each parameter and additional columns for:
        - 'm_tokens_sent': Estimated millions of tokens sent based on the input parameters.
        - 'm_tokens_received': Estimated millions of tokens received based on the input parameters.

    Example
    -------
    >>> employees = np.array([800, 1000])
    >>> calls = np.array([50, 100])
    >>> q_length = np.array([50, 100])
    >>> a_length = np.array([100, 200])
    >>> tokens_per_word = np.array([1.0, 1.5])
    >>> df = make_df_cost(employees, calls, q_length, a_length, tokens_per_word)
    >>> print(df.head())

    Notes
    -----
    This function uses numpy's meshgrid function to create all possible combinations of the input parameters.
    The DataFrame is generated through vectorized operations to ensure efficiency and performance.
    """
    # Generate all combinations of input arrays using numpy's meshgrid
    mesh = np.array(np.meshgrid(employees, calls, q_length, a_length, tokens_per_word))
    combinations = mesh.T.reshape(-1, 5)
    
    # Create DataFrame from combinations
    df = pd.DataFrame(combinations, columns=['employees', 'calls', 'avg_words_per_q', 'avg_words_per_a', 'tokens_per_word'])
    
    # Calculate tokens sent and received
    df['m_tokens_sent'] = ((df['employees'] * df['calls']) * df['avg_words_per_q'] * df['tokens_per_word']) / 1e6
    df['m_tokens_received'] = ((df['employees'] * df['calls']) * df['avg_words_per_a'] * df['tokens_per_word']) / 1e6

    return df

def calculate_total_cost(
    df_costs,
    df_llms,
    models: list[str],
    cost_sent,
    cost_rec,
    tokens_sent_col="m_tokens_sent",
    tokens_received_col="m_tokens_received",
):
    """
    Calculate the cost of questions and answers processed by different models based on the number of tokens sent and received.

    The function modifies the `df_costs` DataFrame in-place by adding new columns that specify the cost for each model's
    questions and answers, as well as the total cost for each model.

    Parameters
    ----------
    df_costs : pandas.DataFrame
        A DataFrame containing the columns specified by `tokens_sent_col` and `tokens_received_col`, which represent
        the millions of tokens sent and received, respectively.
    df_llms : pandas.DataFrame
        A DataFrame containing model cost details, which must include costs per token for sending and receiving,
        as well as a model identifier.
    models : list
        A list of model names as strings. These should match entries in `df_llms`.
    tokens_sent_col : str, optional
        The column name in `df_costs` that contains the number of tokens sent. Default is 'm_tokens_sent'.
    tokens_received_col : str, optional
        The column name in `df_costs` that contains the number of tokens received. Default is 'm_tokens_received'.

    Returns
    -------
    pandas.DataFrame
        The modified `df_costs` DataFrame with additional columns:
        - 'Question cost {model} (USD)': The cost of sending questions for each model.
        - 'Answer cost {model} (USD)': The cost of receiving answers for each model.
        - 'Total cost {model} (USD)': The combined cost of questions and answers for each model.

    Raises
    ------
    KeyError
        If any of the specified columns or models are not found in the respective DataFrames.

    Examples
    --------
    >>> data_costs = {
            'm_tokens_sent': [1.5, 2.0],
            'm_tokens_received': [1.0, 1.5]
        }
    >>> costs_df = pd.DataFrame(data_costs)
    >>> data_llms = {
            'model': ['A', 'B'],
            'cost_sent': [0.01, 0.02],
            'cost_rec': [0.015, 0.025]
        }
    >>> llms_df = pd.DataFrame(data_llms)
    >>> models = ['A', 'B']
    >>> calculate_total_cost(costs_df, llms_df, models)
    """
    # Prevent mutability issues
    df_costs = df_costs.copy()
    df_llms = df_llms.copy()

    for model in models:
        df_costs[f"Question cost {model} (USD)"] = df_costs[
            tokens_sent_col
        ].values * np.array(df_llms.loc[df_llms["modell"] == model, cost_sent])
        df_costs[f"Answer cost {model} (USD)"] = df_costs[
            tokens_received_col
        ].values * np.array(df_llms.loc[df_llms["modell"] == model, cost_rec])
        df_costs[f"Total cost {model} (USD)"] = (
            df_costs[f"Question cost {model} (USD)"]
            + df_costs[f"Answer cost {model} (USD)"]
        )

    return df_costs
