import numpy as np
import time
array = [1,16,7,22,35,62,44,95,12,56]


# 冒泡排序
start1 = time.clock()
def bubbleSort(arr):
    for i in range(1,len(arr)):
        for j in range(0,len(arr)-i):
            if arr[j]>arr[j+1]:
                a = arr[j]
                arr[j] = arr[j+1]
                arr[j+1] = a
    return arr

print(bubbleSort(array))
end1 = time.clock()
print('bubbleSort:%s Seconds'%(end1-start1))


# 选择排序(从大到小)
start2 = time.clock()
def selectionSort(arr):
    for i in range(len(arr)-1):
        maxindex = i
        for j in range(i+1,len(arr)):
            if arr[j]>arr[maxindex]:
                maxindex = j
        if i != maxindex:
            a = arr[i]
            arr[i] = arr[maxindex]
            arr[maxindex] = a
    return arr

print(selectionSort(array))
end2 = time.clock()
print('selectionSort:%s Seconds'%(end2-start2))


# 插入排序
start3 = time.clock()
def insertSort(arr):
    for i in range(len(arr)):
        a = i-1
        b = arr[i]
        while a>=0 and arr[a]>b:
            arr[a+1] = arr[a]
            a = a-1
        arr[a+1] = b
    return arr

print(insertSort(array))
end3 = time.clock()
print('insertSort:%s Seconds'%(end3-start3))


# 归并排序
start4 = time.clock()
def mergeSort(arr):
    import math
    if(len(arr)<2):
        return arr
    middle = math.floor(len(arr)/2)
    left, right = arr[0:middle], arr[middle:]
    return merge(mergeSort(left), mergeSort(right))

def merge(left,right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    while left:
        result.append(left.pop(0))
    while right:
        result.append(right.pop(0))
    return result

print(mergeSort(array))
end4 = time.clock()
print('mergeSort:%s Seconds'%(end4-start4))


# 快速排序
start5 = time.clock()
def quickSort(arr, left=None, right=None):
    left = 0 if not isinstance(left,(int, float)) else left
    right = len(arr)-1 if not isinstance(right,(int, float)) else right
    if left < right:
        partitionIndex = partition(arr, left, right)
        quickSort(arr, left, partitionIndex-1)
        quickSort(arr, partitionIndex+1, right)
    return arr
def partition(arr, left, right):
    pivot = left
    index = pivot+1
    i = index
    while  i <= right:
        if arr[i] < arr[pivot]:
            swap(arr, i, index)
            index+=1
        i+=1
    swap(arr,pivot,index-1)
    return index-1

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

print(quickSort(array))
end5 = time.clock()
print('quickSort:%s Seconds'%(end5-start5))






