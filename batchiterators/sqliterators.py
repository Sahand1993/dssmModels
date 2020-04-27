from abc import ABC, abstractmethod
import mysql.connector
from random import shuffle
from typing import List, Tuple
import numpy as np
import random

from DSSM.batchiterators.batchiterators import DataPointBatch, DataPoint, sample_numbers, DataPointFactory, \
    DEFAULT_IRRELEVANT_SAMPLES, DEFAULT_BATCH_SIZE
from DSSM.batchiterators.reutersrow import ReutersRow, ReutersRowSqlFactory


class SQLIterator(ABC):

    def __init__(self, batch_size = 1, no_of_irrelevant_samples = 4, table: str = None, relevant_col: str = None):
        self._connection = mysql.connector.connect(
            host="localhost",
            user="DSSM",
            passwd="DSSM",
            database="datasets"
        )
        self.cursor = self._connection.cursor()
        self._batch_size = batch_size
        self._no_of_irrelevant_samples = no_of_irrelevant_samples
        self._relevant_col = relevant_col
        self._table = table

        self.current_idx = 0

        if not table:
            raise ValueError

        self.cursor.execute("SELECT COUNT(*) FROM {}".format(table))
        self._max_id = self.cursor.fetchone()[0]
        self._traversal_order = [(table, row_id) for row_id in range(1, self._max_id + 1)]
        shuffle(self._traversal_order)


    def restart(self):
        shuffle(self._traversal_order)
        self.current_idx = 0
        self._connection = mysql.connector.connect(
            host="localhost",
            user="DSSM",
            passwd="DSSM",
            database="datasets"
        )
        self.cursor = self._connection.cursor()


    def __iter__(self):
        return self


    def get_no_of_data_points(self):
        return len(self._traversal_order)


    def __next__(self) -> DataPointBatch: # TODO: Test threaded execution with multiprocessing, to limit stalling because of fetching sql
        if self.current_idx >= len(self._traversal_order):
            self._connection.close()
            self.cursor.close()
            raise StopIteration

        if self.current_idx + self._batch_size - 1 < len(self._traversal_order):
            db_pointers: List[(str, int)] = self._traversal_order[self.current_idx: self.current_idx + self._batch_size]
            self.current_idx += self._batch_size
        else:
            db_pointers: List[(str, int)] = self._traversal_order[self.current_idx: -1]
            self.current_idx = len(self._traversal_order) # TODO check correctness

        samples: List[DataPoint] = self.get_samples(db_pointers)

        return DataPointBatch(samples, no_of_irrelevant_samples=self._no_of_irrelevant_samples)


    def sample_irrelevant(self, id: int, relevant_col: str, table: str) -> List[str]:
        irrelevant_ids = sample_numbers(self._max_id, self._no_of_irrelevant_samples, id)

        self.cursor.execute("SELECT {} FROM {} WHERE id IN ({})"
                       .format(relevant_col, table, ",".join(irrelevant_ids))) # TODO

        incorrect_documents: List[str] = []
        for (documentngrams,) in self.cursor:
            incorrect_documents.append(documentngrams)
        return incorrect_documents


    def get_samples_from_query(self, db_pointers: List[Tuple[str, int]], query: str) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for table, id in db_pointers:
            self.cursor.execute(
                query.format(table, id))

            (questionngrams, documentngrams) = self.cursor.fetchone()
            sample = DataPointFactory.from_data(questionngrams, documentngrams, self.sample_irrelevant(id, self._relevant_col, table))
            samples.append(sample)


        return samples


    def __len__(self):
        rest = len(self._traversal_order) % self._batch_size
        return len(self._traversal_order) // self._batch_size + (1 if rest else 0)


    @abstractmethod
    def get_samples(self, ids: List[Tuple[str, int]]) -> List[DataPoint]:
        pass


class NqSQLIterator(SQLIterator):

    def __init__(self, **kwargs):
        no_of_irrelevant_samples = DEFAULT_IRRELEVANT_SAMPLES
        batch_size = DEFAULT_BATCH_SIZE
        if ("no_of_irrelevant_samples" in kwargs):
            no_of_irrelevant_samples = kwargs["no_of_irrelevant_samples"]
        if ("batch_size" in kwargs):
            batch_size = kwargs["batch_size"]

        super().__init__(batch_size, no_of_irrelevant_samples, "natural_questions", "documentngrams")


    def get_samples(self, ids: List[Tuple[str, int]]) -> List[DataPoint]:
        """
        Take in a list of n ids, and return a list like:

        [data_point1, data_point2, ..., data_pointn]
        :return:
        """
        ids: List[str] = map(lambda t: str(t[1]), ids)
        query = "SELECT id, questionngrams, documentngrams FROM {} WHERE id IN ({})".format(self._table, ",".join(ids)) # TODO: Make one big query instead.
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        samples: List[DataPoint] = list(
            map(lambda dataPoint: DataPointFactory.from_data(
                    dataPoint[1],
                    dataPoint[2],
                    self.sample_irrelevant(dataPoint[0], "documentngrams", "natural_questions")),
                data)
        )  # TODO: Do we need to make list?
        return samples


def nextint_except(start, end, exclude):
    """
    :param start: inclusive
    :param end: exclusive
    :param exclude: not included
    :return:
    """
    while True:
        nextint = np.random.nextint(start, end)
        if (nextint != exclude):
            return nextint


class ReutersSQLIterator(SQLIterator):

    def __init__(self, **kwargs):
        no_of_irrelevant_samples = DEFAULT_IRRELEVANT_SAMPLES
        batch_size = DEFAULT_BATCH_SIZE
        if ("no_of_irrelevant_samples" in kwargs):
            no_of_irrelevant_samples = kwargs["no_of_irrelevant_samples"]
        if ("batch_size" in kwargs):
            batch_size = kwargs["batch_size"]

        super().__init__(batch_size,
                         no_of_irrelevant_samples,
                         ["rcv_articles_with_topic_tags"],
                         "articlengrams")


    def get_samples(self, ids: List[Tuple[str, int]]) -> List[DataPoint]:
        samples: List[DataPoint] = []
        for (table, article_id) in ids:
            article: ReutersRow = ReutersRowSqlFactory.createfrom(article_id)

            sample = DataPointFactory.from_data(article.get_queryarticle_ngrams(), article.get_relevantarticle_ngrams(), self.sample_irrelevant_articles(article))
            samples.append(sample)

        return samples

    def sample_irrelevant_articles(self, article: ReutersRow) -> List[str]: # TODO: Add mechanism that ensures that you only sample irrelevant
        irrelevant: List[str] = []
        while True:
            _, id_irrelevant = random.choice(self._traversal_order)

            otherArticle: ReutersRow = ReutersRowSqlFactory.createfrom(id_irrelevant) # (relevant_col, tags1, tags2, tags3, tags4)

            if not article.isrelevant(otherArticle):
                irrelevant.append(otherArticle.get_queryarticle_ngrams())

            if len(irrelevant) == self._no_of_irrelevant_samples:
                return irrelevant
