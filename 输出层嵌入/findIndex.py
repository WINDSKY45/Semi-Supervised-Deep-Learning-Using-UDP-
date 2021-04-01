import numpy as np
#分类抽样
def equal_proportion_sampling(label, num_classes, size):
#size默认能被num_classes整除，这里不检查能否整除
    feature_number = size // num_classes
    resultIndex = []
    for i in range(num_classes):
        tmp = np.argwhere(label == i)#.flatten()
        index_tmp = np.random.randint(0, tmp.shape[0], size = feature_number)
        if i == 0:
            resultIndex = tmp[index_tmp, 0]
        else:
            resultIndex = np.append(resultIndex,tmp[index_tmp, 0])
    return resultIndex

if __name__ == '__main__':
    label = np.array([[0],[1],[2],[3],[4],[5],[4],[3],[2],[1],[0],[0]])
    num_classes = 5
    size = 5
    print(equal_proportion_sampling(label,num_classes,size))
