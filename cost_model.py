"""
Author: Benedikt Goodman, Division for National Accounts, Statistics Norway
Email: bgo@ssb.no
Created: 22/04/2024
"""
import pandas as pd
import numpy as np

def make_df_cost(employees:int, calls: list[int], q_length: list[int], a_length: list[int], tokens_per_word: float|int):
    """
    Generates a DataFrame that details token usage estimations for a specified number of employees handling
    various numbers of calls, with questions and answers of varying lengths.

    Parameters
    ----------
    employees : int
        The number of employees handling the calls.
    calls : list[int]
        A list of integers, each representing a different number of calls handled.
    q_length : list[int]
        A list of integers where each integer represents the average number of words in a question.
    a_length : list[int]
        A list of integers where each integer represents the average number of words in an answer.
    tokens_per_word : float or int
        The number of tokens generated per word, which is used to estimate token usage.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'employees' : Number of employees.
        - 'calls' : Number of calls.
        - 'avg_words_per_q' : Average number of words per question.
        - 'avg_words_per_a' : Average number of words per answer.
        - 'tokens_per_word' : Number of tokens per word.
        - 'm_tokens_sent' : Estimated millions of tokens sent based on the inputs.
        - 'm_tokens_recieved' : Estimated millions of tokens recieved based on the inputs.
    Raises
    ------
    ValueError
        If any of the input lists are empty, the function will raise a ValueError as it cannot perform
        calculations with empty data ranges.

    Notes
    -----
    The function iteratively calculates the estimated token usage by multiplying the number of calls,
    average words per question, average words per answer, and tokens per word for every combination of
    calls, question lengths, and answer lengths. The calculation assumes all employees handle each scenario.

    Examples
    --------
    >>> employees = 800
    >>> calls = [50, 100, 150]
    >>> q_length = [50, 100]
    >>> a_length = [100, 200]
    >>> tokens_per_word = 1.33
    >>> make_df_cost(employees, calls, q_length, a_length, tokens_per_word)
       employees  calls  avg_words_per_q  avg_words_per_a  tokens_per_word  \
    0        800     50               50              100             1.33   
    1        800     50               50              200             1.33   
    ...
    """ 
    df_list = []
    
    for call in calls:
        for q in q_length:
            for a in a_length:

                data = {
                    'employees': employees,
                    'calls': call,
                    'avg_words_per_q': q,
                    'avg_words_per_a': a,
                    'tokens_per_word': tokens_per_word,
                }

                # Create the DataFrame
                temp_df = pd.DataFrame(data, index=[0])
                temp_df['m_tokens_sent'] = ((800 * call) * q * tokens_per_word) / 10**6
                temp_df['m_tokens_received'] = ((800 * call) * a * tokens_per_word) / 10**6
                
                df_list.append(temp_df)

    return pd.concat(df_list).reset_index(drop=True)

def calculate_total_cost(df_costs, df_llms, models: list[str], cost_sent, cost_rec, tokens_sent_col='m_tokens_sent', tokens_received_col='m_tokens_received'):
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
        df_costs[f'Question cost {model} (USD)'] = df_costs[tokens_sent_col].values * np.array(df_llms.loc[df_llms['modell'] == model, cost_sent])
        df_costs[f'Answer cost {model} (USD)'] = df_costs[tokens_received_col].values * np.array(df_llms.loc[df_llms['modell'] == model, cost_rec])
        df_costs[f'Total cost {model} (USD)'] = df_costs[f'Question cost {model} (USD)'] + df_costs[f'Answer cost {model} (USD)']
        
    return df_costs