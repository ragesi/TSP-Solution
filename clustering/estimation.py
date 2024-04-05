from normalized_cut import QAOACut
from utils.read_dataset import read_dataset
from utils import execute


def estimation(theta, lamda, max_sum):
    lines = read_dataset('ulysses16.tsp', 16)
    points = list()
    for line in lines:
        tmp_point = line.strip().split(' ')
        tmp_point = [float(x) for x in tmp_point]
        tmp_point[0] = int(tmp_point[0])
        points.append([tmp_point[1], tmp_point[2]])

    test = QAOACut(points, theta, lamda, max_sum)
    qc = test.qaoa()
    job = execute.exec_qcircuit(qc, 20000, 'sim', False, None, False)
    output = execute.get_output(job, 'sim')
    return output


if __name__ == '__main__':
    # theta = [1.920699955476545, 13.098043826482622, 4.888403470251021, -7.554532998437712]
    theta = [15.400716302529814, 9.642573763825698, -1.9136677122406152, -0.4362529072196395]
    output = estimation(theta, 6, 0.05)
    print(len(output))

    max_num = 0
    max_bitstring = None
    for item in output.items():
        if item[1] > max_num:
            max_num = item[1]
            max_bitstring = item[0]

    print(max_bitstring[::-1])
    print(max_num)
