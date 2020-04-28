import numpy as np
import os

from typing import List, Iterable, Set
import random
from DSSM.dssm.config import *


DEFAULT_IRRELEVANT_SAMPLES = 4
DEFAULT_BATCH_SIZE = 5

CSV_SEPARATOR = ";"


def createFilteredVocab() -> Set:
    FILTERED_WORD_INDICES = set()
    for line in open(os.environ["THESIS_PROCESSED_DATA_DIR"] + "/vocabulary.txt").readlines():
        wordId, word, count = line.split()
        count = int(count)
        wordId = int(wordId)
        if count > MINFREQ_WORDS:
            FILTERED_WORD_INDICES.add(wordId)

    return FILTERED_WORD_INDICES


FILTERED_WORD_INDICES = createFilteredVocab()


def toints(strings) -> List[int]:
    return list(map(lambda s: int(s), strings))


def readCsvLines(file) -> List[List[str]]:
    return list(map(lambda line: line.split(CSV_SEPARATOR),
        file.readlines()))


def sample_numbers(end: int, size: int, exclude: int) -> List[int]:
    """
    Samples uniformly from 1 to end (inclusive) and excludes exclude.
    :param end:
    :param size:
    :param exclude:
    :return:
    """
    numbers: List[int] = np.random.choice(end, replace=False, size=size + 1) + 1
    numbers: List[int] = filter(lambda number: number != exclude, numbers)
    numbers: List[int] = list(numbers)[:size]
    return numbers


def filterWords(wordIndices: Iterable) -> np.ndarray:
    return np.array(list(filter(lambda wordIndex: wordIndex in FILTERED_WORD_INDICES, wordIndices)))


class DataPoint():

    def __init__(self, _id: int, query_ngrams: np.ndarray, relevant_ngrams: np.ndarray, irrelevant_ngrams: np.ndarray):
        """
        :param query_ngrams: vector of integers
        :param relevant_ngrams: as above
        :param irrelevant_ngrams: matrix of integers, shape is [no_of_irrelevants, None]
        """
        self._id = _id
        self._query_ngrams: np.ndarray = query_ngrams
        self._relevant_ngrams: np.ndarray = relevant_ngrams
        self._irrelevant_ngrams: np.ndarray = irrelevant_ngrams


    def get_query_ngrams(self) -> np.ndarray:
        return self._query_ngrams


    def get_relevant_ngrams(self) -> np.ndarray:
        return self._relevant_ngrams

    def get_irrelevant_ngrams(self) -> np.ndarray:
        return self._irrelevant_ngrams


class DataPointFactory():

    @staticmethod
    def fromNGramsData(_id: int, query_ngrams: str, relevant_ngrams: str, irrelevant_ngrams: List[str]) -> DataPoint:
        return DataPoint(
            _id,
            np.array(toints(query_ngrams.split(","))),
            np.array(toints(relevant_ngrams.split(","))),
            np.array([toints(ngrams.split(",")) for ngrams in irrelevant_ngrams])
        )

    @staticmethod
    def fromWordIndicesData(_id: int, queryWordIndices: str, relevantWordIndices: str, irrelevantWordIndices: List[str]) -> DataPoint:
        return DataPoint(
            _id,
            filterWords(np.array(toints(queryWordIndices.split(",")))),
            filterWords(np.array(toints(relevantWordIndices.split(",")))),
            np.array([filterWords(toints(ngrams.split(","))) for ngrams in irrelevantWordIndices])
        )


class DataPointBatch():

    def __init__(self, data_points: List[DataPoint], no_of_irrelevant_samples = 4):
        self.data_points = data_points
        self._no_of_irrelevant_samples = no_of_irrelevant_samples


    def get_q_indices(self) -> np.ndarray:
        return self.create_batch(map(lambda data_point: data_point.get_query_ngrams(), self.data_points))


    def get_relevant_indices(self) -> np.ndarray:
        return self.create_batch(map(lambda data_point: data_point.get_relevant_ngrams(), self.data_points))


    def create_batch(self, data_vectors) -> np.ndarray:
        indices = np.empty((0, 2), np.int64)
        for i, vector in enumerate(data_vectors):
            new_indices = np.array([[i, index] for index in vector])
            indices = np.concatenate((indices, new_indices))
        return indices

    def get_irrelevant_indices(self) -> List[np.ndarray]:
        irrelevants_batches = [] # [irr1_batch, irr2_batch, irr3_batch]
        for i in range(self._no_of_irrelevant_samples):
            irrelevants_batches.append(self.create_batch(map(lambda data_point: data_point.get_irrelevant_ngrams()[i], self.data_points)))

        return irrelevants_batches


class RandomBatchIterator():
    """
    Randomly calls the __next__() methods of the given iterators.
    """
    def __init__(self, *args):
        """

        :param args: iterators to uniformly sample from.
        """
        self.iterators = list(args)


    def __iter__(self):
        return self


    def __next__(self):
        if self.iterators:
            iterator = random.choice(self.iterators)
            try:
                return iterator.__next__()
            except StopIteration:
                self.iterators.remove(iterator)
                return self.__next__()
        else:
            raise StopIteration


    def restart(self):
        for iterator in self.iterators:
            iterator.restart()


    def __len__(self):
        return sum(map(lambda iterator: len(iterator), self.iterators))