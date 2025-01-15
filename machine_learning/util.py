import base64
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')

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


def get_image_data_from_plot_thread(plot_func, figsize: Optional[Tuple[int, int]] = None) -> bytes:
    img_stream = BytesIO()

    fig = plt.figure(figsize=figsize)
    plot_func()
    plt.savefig(img_stream, format="png")
    fig.clear()
    plt.close()

    img_stream.seek(0)

    img = img_stream.getvalue()
    img_stream.close()
    return img

def get_image_data_from_plot(plot_func, figsize: Optional[Tuple[int, int]] = None) -> bytes:
    """
    Runs the get_image_data_from_plot function in a separate thread.
    :param plot_func: A function that generates a plot.
    :param figsize: Tuple indicating the figure size (width, height).
    :return: Image data as bytes (in PNG format).
    """
    with ThreadPoolExecutor() as executor:
        future = executor.submit(get_image_data_from_plot_thread, plot_func, figsize)
        return future.result()

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
