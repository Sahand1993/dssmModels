from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set
import random
import json

from DSSM.batchiterators.batchiterators import DataPoint, DataPointFactory, readCsvLines, sample_numbers, CSV_SEPARATOR, DataPointBatch


class FileIterator(ABC):

    def __init__(self, batch_size = 5, no_of_irrelevant_samples = 4, encodingType ="NGRAM", path=None):
        if path:
            self._file = open(path)
            self._file.readline()
        self._batch_size = batch_size
        self._no_of_irrelevant_samples = no_of_irrelevant_samples
        self._traversal_order: List[int] = None
        self.current_idx = 0
        self._encodingType = encodingType


    def __next__(self) -> DataPointBatch:
        indices: List[int] = self._traversal_order[self.current_idx: self.current_idx + self._batch_size]
        self.current_idx += self._batch_size
        if len(indices) < self._batch_size:
            raise StopIteration

        return DataPointBatch(self.get_samples(indices), self._no_of_irrelevant_samples)


    @abstractmethod
    def get_samples(self, ids: List[int]) -> List[DataPoint]:
        pass


    @abstractmethod
    def get_irrelevants(self, q_id: int) -> List:
        pass


    @abstractmethod
    def restart(self):
        pass


    def __len__(self):
        return len(self._traversal_order) // self._batch_size


    def __iter__(self):
        return self


    def getNoOfDataPoints(self):
        return (len(self._traversal_order) // self._batch_size) * self._batch_size


class QuoraFileIterator(FileIterator):

    def __init__(self, csvPath:str, batch_size = 5, no_of_irrelevant_samples = 4, encodingType="NGRAM"):
        super().__init__(batch_size,
                         no_of_irrelevant_samples,
                         encodingType,
                         csvPath)
        self._questionIdToDuplicates: Dict[int, List[int]] = dict()
        self._pairIdToQuestionPair: Dict[int, Tuple[str, str]] = dict()
        self._questionIdToIndices: Dict[int, str] = dict()
        self._duplicate_pair_ids: List[int] = []
        self.index_file()
        self._questionIds = list(self._questionIdToDuplicates.keys())
        self._total_samples = len(self._duplicate_pair_ids)
        self._traversal_order = [i for i in self._duplicate_pair_ids] # TODO Implement with ability to provide training or val set but still index all, like in reuters
        random.shuffle(self._traversal_order)


    def restart(self):
        self.current_idx = 0
        self._traversal_order = random.shuffle([i for i in range(len(self._duplicate_pair_ids))])


    def get_samples(self, pairIds: List[int]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for pairId in pairIds:
            q1_id: int = self._pairIdToQuestionPair[pairId][0]
            q2_id: int = self._pairIdToQuestionPair[pairId][1]
            question1_indices: str = self._questionIdToIndices[q1_id]
            question2_indices: str = self._questionIdToIndices[q2_id]
            dataPoint: DataPoint = DataPointFactory.fromNGramsData(pairId, question1_indices, question2_indices, self.get_irrelevants(pairId))
            samples.append(dataPoint)

        return samples


    def get_irrelevants(self, q_id: int) -> List[str]:
        irrelevant_ngrams: List[str] = []
        while True:
            new_q_id: int = random.choice(self._questionIds)
            new_q_duplicates: List[str] = self._questionIdToDuplicates[new_q_id]
            if q_id not in new_q_duplicates:
                irrelevant_ngrams.append(self._questionIdToIndices[new_q_id])

            if (len(irrelevant_ngrams) == self._no_of_irrelevant_samples):
                return irrelevant_ngrams


    def index_file(self):
        for line in self._file.readlines():
            csvValues = line.split(CSV_SEPARATOR)
            pair_id: int = int(csvValues[0])
            q1_id: int = int(csvValues[1])
            q2_id: int = int(csvValues[2])
            if self._encodingType == "NGRAM":
                q1_tokens: List[str] = csvValues[3]
                q2_tokens: List[str] = csvValues[4]
            elif self._encodingType == "WORD":
                q1_tokens: List[str] = csvValues[5]
                q2_tokens: List[str] = csvValues[6]
            else:
                raise ValueError("Wrong value of self._encoding_type")

            is_duplicate = csvValues[7]
            if is_duplicate:
                self._duplicate_pair_ids.append(pair_id)
                try:
                    q1_duplicates: List[int] = self._questionIdToDuplicates[q1_id]
                    if q2_id not in q1_duplicates:
                        q1_duplicates.append(q2_id)
                except KeyError:
                    self._questionIdToDuplicates[q1_id] = [q2_id]

                try:
                    q2_duplicates = self._questionIdToDuplicates[q2_id]
                    if q1_id not in q2_duplicates:
                        q2_duplicates.append(q1_id)
                except KeyError:
                    self._questionIdToDuplicates[q2_id] = [q1_id]

            self._pairIdToQuestionPair[pair_id] = (q1_id, q2_id)

            self._questionIdToIndices[q1_id] = q1_tokens
            self._questionIdToIndices[q2_id] = q2_tokens


class NaturalQuestionsFileIterator(FileIterator):
    def __init__(self, path: str, batch_size = 5, no_of_irrelevant_samples = 4, encodingType="NGRAM"):
        """

        :param path:
        :param batch_size:
        :param no_of_irrelevant_samples:
        :param encodingType: Can be NGRAM or WORD. Determines which document representation will be used.
        """
        super().__init__(batch_size, no_of_irrelevant_samples, encodingType, path)
        self._questionDocumentPairs: List[List[str]] = readCsvLines(self._file)
        self._traversal_order = list(range(len(self._questionDocumentPairs)))
        random.shuffle(self._traversal_order)


    def get_samples(self, indices: List[int]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for idx in indices:
            dataPoint: DataPoint = self.getDataPoint(idx)
            samples.append(dataPoint)

        return samples


    def getDataPoint(self, idx: int) -> DataPoint:
        csvValues = self._questionDocumentPairs[idx]
        _id = csvValues[0]
        irrelevantDocuments: List[str] = self.get_irrelevants(idx)
        if self._encodingType == "NGRAM":
            question, document = csvValues[1], csvValues[2]
            return DataPointFactory.fromNGramsData(_id, question, document, irrelevantDocuments)
        elif self._encodingType == "WORD":
            question, document = csvValues[3], csvValues[4]
            return DataPointFactory.fromWordIndicesData(_id, question, document, irrelevantDocuments)
        else:
            raise ValueError("Incorrect value of self._encoding_type")


    def get_sample(self, idx: int) -> Tuple[str, str]:
        csvValues = self._questionDocumentPairs[idx]
        if self._encodingType == "NGRAM":
            return csvValues[1], csvValues[2]
        elif self._encodingType == "WORD":
            return csvValues[3], csvValues[4]
        else:
            raise ValueError("Incorrect value of self._encoding_type")


    def get_irrelevants(self, idx: int) -> List[str]:
        irrelevantIndices: List[int] \
            = list(map(lambda x: x - 1, sample_numbers(len(self._questionDocumentPairs), self._no_of_irrelevant_samples, idx)))
        irrelevantDocuments: List[str] = []
        for idx in irrelevantIndices:
            _, document = self.get_sample(idx)
            irrelevantDocuments.append(document)
        return irrelevantDocuments


    def restart(self):
        self.current_idx = 0
        random.shuffle(self._traversal_order)


class ReutersFileIterator(FileIterator):
    def __init__(self, dataSetPathJson: str, set = "train", batch_size = 5, no_of_irrelevant_samples = 4, encodingType="NGRAM"):
        """

        :param set: either "train" or "val"
        :param noOfSamples:
        :param batch_size:
        :param no_of_irrelevant_samples:
        :param encodingType:
        """
        super().__init__(batch_size, no_of_irrelevant_samples, encodingType)
        self._idToArticle: Dict[int, Dict] = dict()
        self._tagToId: Dict[str, Set] = dict()
        self.NON_TAG_KEYS = ["queryArticleNGramIndices", "queryArticleWordIndices", "relevantId", "articleId", "id"]
        self._index()
        self._traversal_order: List[int] = self._getArticleIdsFromFile(dataSetPathJson)
        random.shuffle(self._traversal_order)
        self.ARTICLENGRAMS_CSVIDX = None # TODO
        self.TAG_KEYS = [
                "c11",
                "c12",
                "c13",
                "c14",
                "c15",
                "c16",
                "c17",
                "c18",
                "c21",
                "c22",
                "c23",
                "c24",
                "c31",
                "c32",
                "c33",
                "c34",
                "c41",
                "c42",
                "e11",
                "e12",
                "e13",
                "e14",
                "e21",
                "e31",
                "e41",
                "e51",
                "e61",
                "e71",
                "g11",
                "g12",
                "g13",
                "g14",
                "g15",
                "gcrim",
                "gdef",
                "gdip",
                "gdis",
                "gedu",
                "gent",
                "genv",
                "gfas",
                "ghea",
                "gjob",
                "gmil",
                "gobit",
                "godd",
                "gpol",
                "gpro",
                "grel",
                "gsci",
                "gspo",
                "gtour",
                "gvio",
                "gvote",
                "gwea",
                "gwelf",
                "m11",
                "m12",
                "m13",
                "m14",
                "meur"
            ]

    def get_samples(self, ids: List[int]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for _id in ids:
            queryArticle = self._idToArticle[_id]
            relevantArticle = self._idToArticle[queryArticle["relevantId"]]
            irrelevants: List[str] = self.get_irrelevants(_id)
            if self._encodingType == "NGRAM":
                print(queryArticle["queryArticleNGramIndices"])
                print(relevantArticle["queryArticleNGramIndices"])
                print(irrelevants)
                samples.append(DataPointFactory.fromNGramsData(_id,
                                                               queryArticle["queryArticleNGramIndices"],
                                                               relevantArticle["queryArticleNGramIndices"],
                                                               irrelevants))
            elif self._encodingType == "WORD":
                samples.append(DataPointFactory.fromWordIndicesData(_id,
                                                                    queryArticle["queryArticleWordIndices"],
                                                                    relevantArticle["queryArticleWordIndices"],
                                                                    irrelevants))
            else:
                raise ValueError("Incorrect value of self._encodingType")

        return samples


    def restart(self):
        self.current_idx = 0
        random.shuffle(self._traversal_order)


    def get_irrelevants(self, _id):
        irrelevants: List[str] = []
        while True:
            irrelevantId: int = random.choice(self._traversal_order)
            if irrelevantId == _id:
                continue
            if not self.isRelevant(irrelevantId, _id):
                print("was irrelevant")
                if self._encodingType == "NGRAM":
                    irrelevants.append(self._idToArticle[irrelevantId]["queryArticleNGramIndices"])
                else:
                    irrelevants.append(self._idToArticle[irrelevantId]["queryArticleWordIndices"])
            if len(irrelevants) == self._no_of_irrelevant_samples:
                print("returning irrelevants")
                return irrelevants


    def isRelevant(self, id1: int, id2: int) -> bool:
        article1: Dict = self._idToArticle[id1]
        article2: Dict = self._idToArticle[id2]
        article1BoolTagVector: List[bool] = self.getBooleanTagVector(article1)
        article2BoolTagVector: List[bool] = self.getBooleanTagVector(article2)
        print(id1, article1BoolTagVector)
        print(id2, article2BoolTagVector)
        andVector: List[bool] = list(map(lambda tags: tags[0] and tags[1], zip(article1BoolTagVector, article2BoolTagVector)))
        for andResult in andVector:
            if andResult:
                print("return True")
                return True
        print("return false")
        return False


    def getBooleanTagVector(self, article: Dict) -> List[bool]:
        boolVec: List[bool] = []
        for tagKey in self.TAG_KEYS:
            try:
                boolVec.append(bool(article[tagKey]))
            except KeyError:
                boolVec.append(False)
        return boolVec


    def _index(self):
        file = open("/Users/sahandzarrinkoub/School/year5/thesis/DSSM/preprocessed_backup/rcv1/total.json")
        for i, line in enumerate(file.readlines()): # TODO remove enumerate later
            article: Dict = json.loads(line)
            _id = article["id"]
            article.pop("id")
            self._idToArticle[_id] = article

            for tag in self._getTags(article):
                try:
                    self._tagToId[tag].add(_id)
                except KeyError:
                    self._tagToId[tag] = {_id}


    def _getTags(self, article: Dict) -> List[str]:
        tagKeys = filter(lambda key: key not in self.NON_TAG_KEYS, article)
        tags = list(filter(lambda key: article[key] is not False, tagKeys))

        return tags

    def _getArticleIdsFromFile(self, dataSetPathJson) -> List[int]:
        f = open(dataSetPathJson)
        ids = list()
        for line in f:
            jsonObj = json.loads(line)
            ids.append(jsonObj["id"])

        return ids


#nq = NaturalQuestionsFileIterator("/Users/sahandzarrinkoub/School/year5/thesis/DSSM/preprocessed_backup/nq/train.csv", encodingType="WORD")
#quora = QuoraFileIterator("/Users/sahandzarrinkoub/School/year5/thesis/DSSM/preprocessed_backup/quora/train.csv", encodingType="WORD")
# rcv1 = ReutersFileIterator("/Users/sahandzarrinkoub/School/year5/thesis/DSSM/preprocessed_backup/rcv1/validation.json", set = "train", encodingType="NGRAM")
# for batch in rcv1:
#     print(batch)