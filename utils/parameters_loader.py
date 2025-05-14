import json


def save_params(params: dict, file_name: str):
    # Save the dictionary to a file
    with open(f"saved_model_params/{file_name}", "w") as fp:
        json.dump(params, fp)
        print(f"Model params saved to the {file_name}")


def load_params(file_name: str) -> dict:
    """
    Load the dictionary of parameters from a file.

    Parameters
    ----------
    file_name : str
        The name of the file to load the parameters from.

    Returns
    -------
    dict
        The dictionary of loaded parameters.
    """
    with open(f"saved_model_params/{file_name}", "r") as fp:
        params = json.load(fp)
        print(f"Model params loaded from the {file_name}")
    return params


def save_results(params: dict, file_name: str):
    # Save the dictionary to a file
    with open(f"results/{file_name}", "w") as fp:
        json.dump(params, fp)
        print(f"Results saved to the {file_name}")
