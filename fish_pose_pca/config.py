import json

_config = None


class NoConfigLoadedException(Exception):
    pass


class ConfigMissesKeysException(Exception):
    pass


_MASK_PATH = "mask_path"
_MASK_THRESHOLD = "mask_threshold"
_MIN_CONTOUR_POINTS = "min_contour_points"
_CONTOUR_DATA_PATH = "contour_data_path"


def _access_config(key):
    if not _config:
        raise NoConfigLoadedException
    try:
        return _config[key]
    except KeyError:
        raise ConfigMissesKeysException


def load_config(config_path):
    """
    load configuration json file
    :param config_path: str or Path
    """
    config_path = str(config_path)
    global _config
    with open(config_path) as c:
        _config = json.load(c)


def create_config(config_path, config_dict=None):
    """
    create a default config json file
    :param config_dict: optional dict to create config file, if config_dict is None, a default config is created.
    :param config_path: str or Path
    """
    if not config_dict:
        config_dict = {
            _MASK_PATH: "",
            _MASK_THRESHOLD: 0.5,
            _MIN_CONTOUR_POINTS: 0,
            _CONTOUR_DATA_PATH: ""
        }
    config_path = str(config_path)
    with open(config_path, mode="w") as c:
        json.dump(config_dict, c)


def get_mask_path():
    return _access_config(_MASK_PATH)


def get_mask_threshold():
    return _access_config(_MASK_THRESHOLD)


def get_min_contour_points():
    return _access_config(_MIN_CONTOUR_POINTS)


def get_contour_data_path():
    return _access_config(_CONTOUR_DATA_PATH)
