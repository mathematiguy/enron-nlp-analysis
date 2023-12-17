import math
import concurrent.futures
import multiprocessing
import pandas as pd
from functools import partial
from tqdm.auto import tqdm


def process_chunk(chunk, func):
    return [func(item) for item in chunk]


def parallel_apply(input_list, func, cores=None, n_series=0):
    """
    Applies a function to each item of a list or pandas Series in parallel, optionally processing the first n items sequentially.

    Args:
        input_list (list or pd.Series): The list or pandas Series of items to which the function will be applied.
        func (function): The function to apply to each item. This function should take a single argument and return a result.
        cores (int, optional): The number of CPU cores to use for parallel processing. If None, uses the number of available cores.
        n_series (int, optional): The number of items to process sequentially before parallel processing. Default is 0.

    Returns:
        list or pd.Series: A list or pandas Series (matching the input type) of results with the function applied to each item.

    This function processes the first `n_series` items of the input list/series sequentially, and then processes the
    remaining items in parallel using multiprocessing. The list is divided into chunks, each processed by a separate
    process. If the input is a pandas Series, the output will also be a Series with the same index.
    """

    if cores is None:
        cores = multiprocessing.cpu_count()

    # Check if input is a pandas Series and store the index
    is_series = isinstance(input_list, pd.Series)
    if is_series:
        index = input_list.index

    results = []
    # Process first n_series items in series
    for item in input_list[:n_series]:
        results.append(func(item))

    # Process remaining items in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        # Split the list into chunks for each core
        chunk_size = len(input_list[n_series:]) // cores + 1
        future_to_chunk = {
            executor.submit(
                partial(process_chunk, func=func),
                input_list[n_series:][i : i + chunk_size],
            ): i
            for i in range(0, len(input_list[n_series:]), chunk_size)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_chunk),
            total=len(future_to_chunk),
            desc="Processing",
        ):
            results.extend(future.result())

    # Convert back to Series if the input was a Series
    if is_series:
        return pd.Series(results, index=index)

    return results


def parallel_batch_apply(input_list, batch_func, batch_size=None, cores=None):
    """
    Applies a function to batches of items in a list in parallel.

    Args:
        input_list (list): The list of items to process.
        batch_func (function): The function to apply to each batch. This function should take a list (batch) and return a list.
        batch_size (int, optional): The size of each batch. If None, it's calculated based on the number of cores.
        cores (int, optional): The number of CPU cores to use. Defaults to the number of available cores.

    Returns:
        list: The combined results from processing each batch.
    """
    if cores is None:
        cores = multiprocessing.cpu_count()

    # Calculate the batch size
    if batch_size is None:
        batch_size = math.ceil(len(input_list) / cores)

    # Split the list into batches
    batches = [
        input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)
    ]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        # Submit all batches for processing
        future_to_batch = {
            executor.submit(batch_func, batch): i for i, batch in enumerate(batches)
        }

        # Collect results as batches are completed
        for future in tqdm(
            concurrent.futures.as_completed(future_to_batch),
            total=len(batches),
            desc="Processing Batches",
        ):
            results.extend(future.result())

    return results
