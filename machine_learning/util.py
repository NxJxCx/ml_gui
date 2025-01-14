import base64
import secrets
from io import BytesIO
from typing import Any
from weakref import WeakMethod

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


def encode_base64(byte_data: bytes) -> str:
    """
    Encode bytes to a Base64 string.

    :param byte_data: The byte data to be encoded.
    :return: A Base64 encoded string representation of the byte data.
    """
    return base64.b64encode(byte_data).decode("utf-8")


def decode_base64(encoded_data: str) -> bytes:
    """
    Decode a Base64 string back to bytes.

    :param encoded_data: The Base64 encoded string to be decoded.
    :return: The decoded byte data.
    """
    return base64.b64decode(encoded_data)


def get_image_data_from_plot(plot_func) -> bytes:
    """
    Captures image data from a plot function.
    :param plot_func: A function that generates a plot (e.g., `plot_tree`, `disp.plot`, etc.)
    :return: Image data as bytes (in PNG format).
    """
    # Create a BytesIO object to save the plot as image data
    img_stream = BytesIO()

    # Generate the plot using the provided function
    pf = WeakMethod(plot_func)
    func = pf()
    if func:
        func()

    # Save the plot to the image stream in PNG format
    plt.savefig(img_stream, format="png")

    # Rewind the buffer to the beginning for reading
    img_stream.seek(0)

    # Return the image data as bytes
    return img_stream.getvalue()


def generate_session_id():
    return secrets.token_hex(12)


def recursive_json_compatible(elem: Any):
    if isinstance(elem, np.ndarray):
        return elem.tolist()
    elif isinstance(elem, pd.DataFrame):
        return elem.to_dict()
    elif isinstance(elem, pd.Series):
        if elem.index.equals(pd.Index(range(len(elem)))):
            return elem.tolist()
        else:
            return elem.to_dict()
    elif type(elem) is int or type(elem) is float or type(elem) is bool:
        return elem
    elif type(elem) is bytes:
        return encode_base64(elem)
    elif type(elem) is list or type(elem) is tuple or type(elem) is set:
        try:
            result = json.dumps(elem)
        except:
            result = []
            for el in elem:
                result.append(recursive_json_compatible(el))
        return result
    elif type(elem) is dict:
        try:
            result = json.dumps(elem)
        except:
            result = {}
            for l, el in elem.items():
                result[l] = recursive_json_compatible(el)
        return result
    else:
        return str(elem)


def map_list_json_compatible(data: Any):
    result = []
    for elem in data:
        result.append(recursive_json_compatible(elem))
    return result
