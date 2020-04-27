from __future__ import annotations
from typing import List, Tuple, Set
import mysql.connector


def share_one_member(a: Set[str], b: Set[str]):
    for key in a:
        if (key in b):
            return True

    return False


class ReutersRow():
    def __init__(self, article_id: int, queryarticle_ngrams: str, relevantarticle_ngrams: str, tag_groups: List[str], boolean_tags: List[bool]):
        self._article_id: int = article_id
        self._queryarticle_ngrams: str = queryarticle_ngrams
        self._relevantarticle_ngrams: str = relevantarticle_ngrams
        self._tag_groups: List[str] = tag_groups
        self._boolean_tags: List[bool] = boolean_tags


    def isrelevant(self, other: ReutersRow):
        commonTagGroups = list(filter(lambda tagGroup, tagGroupOther: tagGroup and tagGroup, zip(self._tag_groups, other._tag_groups)))
        commonBooleanTags = list(filter(lambda boolean, otherBoolean: boolean and otherBoolean, zip(self._boolean_tags, other._boolean_tags)))
        return (len(commonTagGroups) > 0) or (len(commonBooleanTags) > 0)


    def getid(self) -> int:
        return self._article_id


    def get_queryarticle_ngrams(self) -> str:
        return self._queryarticle_ngrams


    def get_relevantarticle_ngrams(self) -> str:
        return self._relevantarticle_ngrams


def maketagsets(tags: Tuple[str]) -> Tuple[Set[str], ...]:
    result = ()
    for current_tags in tags:
        if current_tags:
            result += (set(current_tags.split(",")),)
        else:
            result += (set(),)

    return result


class ReutersRowFactory():
    @staticmethod
    def createfrom(features: List[str]):
        features
        # TODO: Separate and call ReutersRow()


class ReutersRowSqlFactory():
    _connection = mysql.connector.connect(
        host="localhost",
        user="DSSM",
        passwd="DSSM",
        database="datasets"
    )
    _cursor = _connection.cursor()

    @staticmethod
    def createfrom(row_id: int):
        ReutersRowSqlFactory._cursor.execute("SELECT query_articlengrams, relevant_article_id, c11, c12, c13, c14, c15, c16, c17, c18, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, e11, e12, e13, e14, e21, e31, e41, e51, e61, e71, g11, g12, g13, g14, g15, m11, m12, m13, m14, gcrim, gdef, gdip, gdis, gedu, gent, genv, gfas, ghea, gjob, gmil, gobit, godd, gpol, gpro, grel, gsci, gspo, gtour, gvio, gvote, gwea, gwelf, meur FROM rcv_articles_with_topic_tags where id={}".format(row_id))
        result = ReutersRowSqlFactory._cursor.fetchone()
        query_articlengrams = result[0]
        relevant_article_id = result[1]
        ReutersRowSqlFactory._cursor.execute(
            "SELECT query_articlengrams FROM rcv_articles_with_topic_tags WHERE id={}".format(relevant_article_id))
        relevant_articlengrams = ReutersRowSqlFactory._cursor.fetchone()[0]
        return ReutersRow(row_id, query_articlengrams, relevant_articlengrams, result[2:39], result[39:])

    @staticmethod
    def close():
        ReutersRowSqlFactory._cursor.close()