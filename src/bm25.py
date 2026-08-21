"""
bm25.py
=======

Minimal, dependency-free BM25 (Okapi) implementation.

We implement this from scratch rather than pulling in rank_bm25 so the
sparse baseline has zero extra dependencies beyond numpy, which the
project already requires for the dense pipeline. This also keeps the
scoring fully transparent for the writeup.

Standard Okapi BM25:

    score(q, d) = sum over query terms t of
        IDF(t) * ( f(t, d) * (k1 + 1) )
                  -----------------------------------------
                  ( f(t, d) + k1 * (1 - b + b * |d| / avgdl) )

    IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )

        N     = number of documents in the collection
        n(t)  = number of documents containing term t
        f(t,d)= raw term frequency of t in document d
        |d|   = length of document d in tokens
        avgdl = average document length across the collection
"""

import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


class BM25Okapi:

    def __init__(self, documents, k1=1.5, b=0.75):
        """
        documents: list of raw text strings (one per chunk). Index
        order is preserved and used as the document id returned by
        get_scores()/top_k().
        """

        self.k1 = k1
        self.b = b

        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]

        self.n_docs = len(documents)

        self.avgdl = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0
        )

        # Term frequency per document
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        # Document frequency per term (how many docs contain term t)
        doc_freq = Counter()

        for tf in self.term_freqs:
            for term in tf.keys():
                doc_freq[term] += 1

        # Precompute IDF for every term seen in the collection
        self.idf = {}

        for term, freq in doc_freq.items():

            self.idf[term] = math.log(
                (self.n_docs - freq + 0.5) / (freq + 0.5) + 1
            )

    def get_scores(self, query):
        """Return a list of BM25 scores, one per document, in index order."""

        query_terms = tokenize(query)

        scores = [0.0] * self.n_docs

        for term in query_terms:

            idf = self.idf.get(term)

            if idf is None:
                # Term never appears in the collection -> no contribution
                continue

            for doc_id in range(self.n_docs):

                freq = self.term_freqs[doc_id].get(term, 0)

                if freq == 0:
                    continue

                doc_len = self.doc_lengths[doc_id]

                denom = freq + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )

                scores[doc_id] += idf * (freq * (self.k1 + 1)) / denom

        return scores

    def top_k(self, query, k=5):
        """Return [(doc_id, score), ...] sorted by score descending."""

        scores = self.get_scores(query)

        ranked = sorted(
            range(self.n_docs), key=lambda i: scores[i], reverse=True
        )

        return [(doc_id, scores[doc_id]) for doc_id in ranked[:k]]
