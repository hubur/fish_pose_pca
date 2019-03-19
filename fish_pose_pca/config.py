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
_CONTOUR_START_NORMALIZATION = "contour_start_normalization"
_TRANSLATION_METHOD = "translation_method"
_ROTATION_METHOD = "rotation_method"
_SUBFISH_SIZE = "subfish_size"
_SUBFISH_METHOD = "subfish_method"
_COORDINATE_SYSTEM = "coordinate_system"


class ContourStart:
    MOST_DISTANT_FROM_CENTER_OF_MASS = "most_distant_from_center_of_mass"


class TranslationMethod:
    CENTER_OF_MASS_ON_ZERO = "center_of_mass_on_zero"
    CENTER_OF_MASS_ON_X100 = "center_of_mass_on_x100"

class RotationMethod:
    MOST_DISTANT_POINT_AND_CENTER_ON_LINE = "most_distant_point_and_center_on_line"


class SubFishMethod:
    RANDOM_SUBSET = "random_subset"
    QUADRATIC_INTERPOLATE = "quadratic_interpolate"


class CoordinateSystem:
    CARTESIAN_SYSTEM = "cartesian_system"
    POLAR_SYSTEM = "polar_system"


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
            _CONTOUR_DATA_PATH: "",
            _CONTOUR_START_NORMALIZATION: ContourStart.MOST_DISTANT_FROM_CENTER_OF_MASS,
            _TRANSLATION_METHOD: TranslationMethod.CENTER_OF_MASS_ON_ZERO,
            _ROTATION_METHOD: RotationMethod.MOST_DISTANT_POINT_AND_CENTER_ON_LINE,
            _SUBFISH_METHOD: SubFishMethod.RANDOM_SUBSET,
            _SUBFISH_SIZE: 45,
            _COORDINATE_SYSTEM: CoordinateSystem.CARTESIAN_SYSTEM
        }
    config_path = str(config_path)
    with open(config_path, mode="w") as c:
        json.dump(config_dict, c, indent=True)


def get_mask_path():
    return _access_config(_MASK_PATH)


def get_mask_threshold():
    return _access_config(_MASK_THRESHOLD)


def get_min_contour_points():
    return _access_config(_MIN_CONTOUR_POINTS)


def get_contour_data_path():
    return _access_config(_CONTOUR_DATA_PATH)


def get_contour_start():
    return _access_config(_CONTOUR_START_NORMALIZATION)


def get_translation_method():
    return _access_config(_TRANSLATION_METHOD)


def get_rotation_method():
    return _access_config(_ROTATION_METHOD)


def get_subfish_method():
    return _access_config(_SUBFISH_METHOD)


def get_subfish_size():
    return _access_config(_SUBFISH_SIZE)


def get_coordinate_system():
    return _access_config(_COORDINATE_SYSTEM)
