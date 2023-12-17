import concurrent.futures
import multiprocessing
import pandas as pd
from functools import partial
from tqdm.auto import tqdm


def process_chunk(chunk, func):
    return [func(item) for item in chunk]


def parallel_apply(input_list, func, cores=None, n_series=0):
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
