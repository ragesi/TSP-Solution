def validate_inputs(point_num, points):
    """
    determining the input of QUOTA is legal
    :param point_num: int
    :param points: list
    """
    try:
        if point_num <= 0:
            raise ValueError("point_num must be greater than 0")
        # if point_num != len(points):
        #     raise ValueError("point_num must be equal to length of points")
    except ValueError as e:
        print(f"catch the value error:{e}")
        exit(1)


def trans_to_cycle(points):
    """
    transforming the input of QUOTA to the type of cycle
    """
    points.append(points[0])
    return points
