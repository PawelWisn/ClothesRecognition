import pickle as pkl
import numpy as np


def predict(x):
    """
    Function takes images as the argument. They are stored in the matrix X (NxD).
    Function returns a vector y (Nx1), where each element of the vector is a class numer {0, ..., 9} associated with recognized type of cloth.
    :param x: matrix NxD
    :return: vector Nx1
    """
    valPicsIntNoNoise = np.load("validationsetpics.npy", 'r', allow_pickle=True)
    valLabels = np.load("validationsetlabels.npy", 'r', allow_pickle=True)

    k = 3
    threshold = 0.31

    picsNoNoise = np.array(list(map(lambda pic: eliminateNoise(pic), x)))

    picsIntNoNoise = np.array(
        list(map(lambda pic: np.array([cell > threshold for cell in pic], dtype=int), picsNoNoise)))

    hammDist = hammingDistance(valPicsIntNoNoise, picsIntNoNoise)

    sortedLabels = sortLabels(hammDist, valLabels)

    answers = getAnswers(sortedLabels, k)

    return np.asarray(answers).reshape(-1, 1)


def hammingDistance(compares, images):
    # idea from: https://stackoverflow.com/questions/44092101/hamming-distance-between-two-large-ndarrays-optimization-of-large-array-multipl
    comparesT = np.transpose(compares)
    notImages = np.subtract(np.ones(shape=(images.shape[0], images.shape[1])), images)
    notCompares = np.subtract(np.ones(shape=(comparesT.shape[0],
                                               comparesT.shape[1])), comparesT)
    return images @ notCompares + notImages @ comparesT


def sortLabels(hammingDist, labels):
    s = np.argsort(hammingDist, kind='quicksort')
    return labels[s]


def getAnswers(sortedLabels, k):
    labels = [i for i in range(10)]
    answers = []
    for row in sortedLabels:
        temp = []
        for label in labels:
            segment = row[:k]
            temp.append(np.sum(list(map(lambda neigh: neigh == label, segment))) / k)
        answers.append(np.argmax(temp))

    return answers


def eliminateNoise(pic, pic_side=36, frame_side=28):
    brightTres = 0.65
    brightVal = 0.05
    noiseTres = 0.33
    noiseVal = 0.22

    def cellVal(cell):
        if cell > brightTres:
            return brightVal
        elif cell > noiseTres:
            return noiseVal
        else:
            return cell

    minNoise = np.inf
    bestX, bestY = 0, 0
    pic = pic.reshape(pic_side, pic_side)
    for x in range(pic_side - frame_side):
        for y in range(pic_side - frame_side):
            noise = 0
            noise += sum(map(lambda cell: cellVal(cell), pic[x + frame_side][y:y + frame_side]))  # hor bot
            noise += sum(map(lambda cell: cellVal(cell), pic[x:, y + frame_side][:frame_side]))  # ver right
            noise += sum(map(lambda cell: cellVal(cell), pic[x:, x][:frame_side]))  # ver left
            noise += sum(map(lambda cell: cellVal(cell), pic[x][y:y + frame_side]))  # hor top

            if minNoise > noise:
                minNoise = noise
                bestX, bestY = x, y

    outPic = []
    for x in range(bestX, bestX + frame_side):
        outPic.append(pic[x][bestY:bestY + frame_side])
    return np.array(outPic).flatten()
