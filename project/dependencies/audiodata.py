from typing import Literal, Union, List
import numpy as np

class AudioDataPoint:
    def __init__(self, timestamp, pitch, volume, mfcc):
        self.timestamp = timestamp
        self.pitch = pitch
        self.volume = volume
        self.mfcc = mfcc

class AudioDataSet(list):

    def __init__(self, *data):
        self.ranges = []
        super().__init__(*data)

    def add_frame(self, timestamp, pitch, volume, mfcc):
        super().append(AudioDataPoint(timestamp, pitch, volume, mfcc))

    def get_frame(self, timestamp):
        return self[self.indexOf(timestamp)]

    def __calculate_deviation(self, center, actual):
        return center-actual

    def indexOf(self, timestamp):
        for i in range(len(self)):
            if self[i].timestamp == timestamp:
                return i
            

    def closest_point(self, time, snapping : Literal["floor", "near", "ceil"] = "near"):       
        if len(self) == 0:
            raise ValueError("Dataset is empty. Cannot find the closest point.")
        
        lowerThresh = -1
        ret = 0

        for i in range(0, len(self)):
            if self[i].timestamp <= time:
                lowerThresh = i
            else:
                break

        if lowerThresh == -1:  # All timestamps are greater
            lowerThresh = 0
        elif lowerThresh == len(self) - 1 and self[lowerThresh].timestamp <= time:  # All timestamps are smaller
            if snapping == "floor" or snapping == "near":
                return lowerThresh, self.__calculate_deviation(self[lowerThresh].timestamp, time)
            elif snapping == "ceil":
                return lowerThresh + 1, float("inf")  # Ceil does not exist for `time` after the largest point
        
        if snapping == "floor":
            return lowerThresh, self.__calculate_deviation(self[lowerThresh].timestamp, time)

        elif snapping == "ceil":
            if lowerThresh + 1 < len(self):
                return lowerThresh + 1, self.__calculate_deviation(self[lowerThresh + 1].timestamp, time)
            else:
                raise ValueError("Ceiling value does not exist.")
        
        elif snapping == "near":
            # Compare the deviations between `lowerThresh` and `lowerThresh + 1`
            if lowerThresh + 1 < len(self):
                lower_dev = abs(self.__calculate_deviation(self[lowerThresh].timestamp, time))
                upper_dev = abs(self.__calculate_deviation(self[lowerThresh + 1].timestamp, time))
                if lower_dev <= upper_dev:
                    return lowerThresh, self.__calculate_deviation(self[lowerThresh].timestamp, time)
                else:
                    return lowerThresh + 1, self.__calculate_deviation(self[lowerThresh + 1].timestamp, time)
            else:
                return lowerThresh, self.__calculate_deviation(self[lowerThresh].timestamp, time)
        else:
            raise ValueError("Invalid 'snapping' type. Use 'floor', 'near', or 'ceil'.")
        
    def max(self, type: Literal["pitch", "volume"], indexes : List[int]):
        max = getattr(self[indexes[0]], type)
        ret = self[indexes[0]]
        for i in indexes:
            if max < getattr(self[i], type):
                max = getattr(self[i], type)
                ret = self[i]
        return ret

    def min(self, type: Literal["pitch", "volume"], indexes : List[int]):
        min = getattr(self[indexes[0]], type)
        ret = self[indexes[0]]
        for i in indexes:
            if min > getattr(self[i], type):
                min = getattr(self[i], type)
                ret = self[i]
        return ret
    
    def avg(self, type: Literal["pitch", "volume"], indexes):
        sum = 0
        for i in indexes:
            sum += getattr(self[i], type)
        return sum/len(indexes)
    
    def pitch_zero_point(self, indexes: List[int]):
        max = self.max('pitch', indexes)
        min = self.min('pitch', indexes)
        return np.mean([max, min])

    def pauseIndexes(self, threshold, indexes=None):
        ret = []
        if indexes is None:
            indexes = self.indexes()
        for i in indexes:
            if self[i].volume < threshold:
                ret.append(i)
        return ret

    def add_range(self, start, end, dialogue=""):
        self.ranges.append(AudioDataRange(self, start, end, dialogue=dialogue))

    def indexes(self):
        return range(len(self))



class AudioDataRange(AudioDataSet):

    def __init__(self, dataset: Union[AudioDataSet, "AudioDataRange"], start:int, end:int, dialogue=""):
        self.parent = dataset
        self.dialogue = dialogue
        self.start = start
        self.end = end
        super().__init__(dataset[start:end])
        self.lineageCount = self.parent.lineageCount+1 if type(self.parent) is AudioDataRange else 1

    def getParent(self, levels=1):
        return self.parent.getParent(levels-1) if levels > 1 and type(self) is AudioDataRange else self.parent

def indexesOfTalking(dataset: AudioDataSet):
    initial = []
    for i in dataset.ranges:
        initial.extend(range(i.start, i.end))
    return initial

def indexesOfBackground(dataset: AudioDataSet):
    return [x for x in range(len(dataset)) if x not in indexesOfTalking(dataset)]


def joinRanges(arr: List["AudioDataRange"]):
    myList = AudioDataSet([])
    for i, val in enumerate(arr):
        myList.extend(val)
        myList.add_range(myList.indexOf(val[0].timestamp), myList.indexOf(val[len(val)-1].timestamp), val.dialogue)
    
    return AudioDataSet(myList)