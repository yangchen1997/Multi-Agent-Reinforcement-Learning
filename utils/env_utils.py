from numpy import ndarray

from utils.config_utils import ConfigObjectFactory


def map_value(data: ndarray) -> tuple:
    """
        将agent现有的坐标进行映射，暂时假设原坐标区间为[-3,3], 需要映射到[0,grid_size]区间
        :param data: 原坐标
        :return: 映射坐标
    """
    grid_size = ConfigObjectFactory.get_environment_config().grid_size
    target_min = 0
    target_max = grid_size - 1
    pos_x, pos_y = clip_pos((float(data[0]), float(data[1])), min_value=-1.5, max_value=1.5)
    map_pos_x = target_min + (target_max - target_min) / (1.5 - -1.5) * (pos_x - -1.5)
    map_pos_y = target_min + (target_max - target_min) / (1.5 - -1.5) * (pos_y - -1.5)
    return map_pos_x, map_pos_y


def clip_pos(old_pos: tuple, min_value: float, max_value: float) -> tuple:
    """
        对坐标进行裁剪，以免超出grid_input的范围
        :param min_value:
        :param max_value:
        :param old_pos: 待剪裁的坐标
        :return: 新坐标
    """
    if len(old_pos) != 2:
        raise ValueError(
            'Expecting a list of length 2 as input, but now the length of the input is {}.'.format(len(old_pos)))
    x = old_pos[0]
    y = old_pos[1]
    if x < min_value:
        x = min_value
    if x > max_value - 1:
        x = max_value - 1
    if y < min_value:
        y = min_value
    if y > max_value - 1:
        y = max_value - 1
    return x, y


def get_approximate_pos(pos: tuple) -> tuple:
    """
        根据现有的位置，计算近似位置(将小数变成整数)
        :param pos:
        :return:
    """
    if len(pos) != 2:
        raise ValueError(
            'Expecting a list of length 2 as input, but now the length of the input is {}.'.format(len(pos)))
    grid_size = ConfigObjectFactory.get_environment_config().grid_size
    approximate_pos = int(round(pos[0], 0)), int(round(pos[1], 0))
    return clip_pos(approximate_pos, min_value=0, max_value=grid_size)


def recomput_approximate_pos(pos_dict: dict, approximate_pos: tuple, pos: tuple) -> tuple:
    """
        多个agent的近似位置发生冲突，重新计算近似位置
        :param pos_dict:
        :param approximate_pos:
        :param pos:
        :return:
    """
    if not pos_dict:
        raise ValueError('pos_dict is null or length of pos_dict is 0')
    grid_size = ConfigObjectFactory.get_environment_config().grid_size
    exist_pos = pos_dict[approximate_pos]
    gap_x = abs(exist_pos[0] - pos[0])
    gap_y = abs(exist_pos[1] - pos[1])
    approximate_pos_x = approximate_pos[0]
    approximate_pos_y = approximate_pos[1]
    if gap_x <= gap_y:
        if exist_pos[0] >= pos[0]:
            approximate_pos_x -= 1
        else:
            approximate_pos_x += 1
    else:
        if exist_pos[1] >= pos[1]:
            approximate_pos_y -= 1
        else:
            approximate_pos_y += 1
    return clip_pos((approximate_pos_x, approximate_pos_y), min_value=0, max_value=grid_size)
