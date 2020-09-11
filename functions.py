import numpy as np

def splitTo4(string):
    n = 4
    res = [(string[i:i + n]) for i in range(0, len(string), n)]
    return res


def h2i(hexstring):
    # c = []
    ba = bytes.fromhex(hexstring)
    result = int.from_bytes(ba, byteorder='little', signed=True)
    # c.append(result)
    # return c
    return result


def complex2real(array):
    real_list = []
    for elem in array:
        real_list.append(abs(elem))
    return(real_list)


def normalize(array):
    array_normalized = []
    for elem in array:
        temp = (elem - min(array))/(max(array)-min(array))
        array_normalized.append((temp))
    return array_normalized


def locMax(arr):
    mx = []
    len = arr.__len__()
    #print("Length of the array passes is {}".format(len))
    # first element
    if (arr[0] > arr[1]):
        mx.append(arr[0])
    elif (arr[0] < arr[1]):
        mx.append(0)

    for i in range(1, len - 1):
        # if(arr[i-1] < arr[i] > arr[i + 1]):
        # mx.append(i)
        if (arr[i] > arr[i - 1] and arr[i] > arr[i + 1]):
            mx.append(arr[i])
        else:
            mx.append(0)

    # last element
    if (arr[-1] > arr[-2]):
        mx.append(arr[-1])
    else:
        mx.append(0)

    return mx


def list2tuple(list):
    return tuple(list)

def eucdist(x,y):
    return np.linalg.norm(x-y)


"""
def localmaxima(a, n):
    lmaxlist = []
    for i in range(1, n - 1):
        # check if element is greater than both the neighbours,
        # the first and last element won't be considered
        if(a[i] > a[i - 1] and a[i] > a[i + 1]):
            lmaxlist.append(a[i])
    abs_lmaxlist = []
    for elem in lmaxlist:
        abs_lmaxlist.append(abs(elem))
    return abs_lmaxlist"""

"""
def findLocalMaxima(arr, n):

    mx = []
    #check 1st value
    if (arr[0] > arr[1]):
        mx.append(arr[0])
    mx.append(0)
    #check 2nd to last but one value
    for i in range(1, n - 1):
        if (arr[i - 1] < arr[i] > arr[i + 1]):
            mx.append(arr[i])
        else:
            mx.append(0)
    #check last  value
    if (arr[-1] > arr[-2]):
        mx.append(arr[-1])
    mx.append(0)

    return mx
"""

"""
def localmaxima(a, n):
    lmaxlist = []
    for i in range(1, n - 1):
        # check if element is greater than both the neighbours,
        # the first and last element won't be considered
        if(a[i] > a[i - 1] and a[i] > a[i + 1]):
            lmaxlist.append(a[i])
    abs_lmaxlist = []
    for elem in lmaxlist:
        abs_lmaxlist.append(abs(elem))
    return abs_lmaxlist
"""





