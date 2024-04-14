def read_dataset(file_name, scale):
    file_path = '../dataset/' + str(scale) + '/' + file_name
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # lines = [line[:-1] for line in lines]

    # finding the start of points
    i = 0
    while i < len(lines):
        if lines[i][0].isdigit():
            break
        i += 1
    lines = lines[i:]

    # finding the end of points
    i = len(lines) - 1
    while i >= 0:
        if lines[i][0] != 'E' and lines[i][0] != '-':
            break
        i -= 1
    lines = lines[:(i + 1)]
    return lines


if __name__ == '__main__':
    data = read_dataset('dj38.opt.tour', 38)
    print(len(data))
    print(data)
