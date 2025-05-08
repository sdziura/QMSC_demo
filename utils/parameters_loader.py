import json


def save_params(params: dict, file_name: str):
    # Save the dictionary to a file
    with open(f"saved_model_params/{file_name}", "w") as fp:
        json.dump(params, fp)
        print(f"Model params saved to the {file_name}")
