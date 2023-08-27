import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import spacy
from nltk import SnowballStemmer


class BasePreprocessor(ABC):
    @abstractmethod
    def __call__(self, text: str) -> str:
        raise NotImplementedError


class WhitespaceRemovePreprocessor(BasePreprocessor):
    def __call__(self, text: str) -> str:
        """
        Basic cleaning :
            1. remove any hyphen followed by one or more whitespace
            2. removes double white spaces between words
            3. removes beginning and trailing whitespace in a string
        :param text: input string
        :return: str
        """
        text = re.sub(r"-\s+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s$|^\s", "", text)
        return text


class MakeLowerCasePreprocessor(BasePreprocessor):
    def __call__(self, text: str) -> str:
        """
        Makes all string lower case
        """
        return text.lower()


class StemPreprocessor(BasePreprocessor):
    def __init__(self, language: str = "english"):
        self.stemmer = SnowballStemmer(language)

    def __call__(self, text: str) -> str:
        """
        Stemming operation using nltk
        """
        return " ".join([self.stemmer.stem(word) for word in text.split()])


class LemmaPreprocessor(BasePreprocessor):
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.spacy_model = spacy.load(model_name)

    def __call__(self, text: str) -> str:
        """
        Lemmatization operation using nltk
        """
        return " ".join([token.lemma_ for token in self.spacy_model(text)])


# TODO: Add more preprocessing steps


class StackedPreprocessor(BasePreprocessor):
    def __init__(self, preprocessors: Optional[Dict[str, Optional[Dict]]] = None):
        if preprocessors is None:
            preprocessors = {}

        self.preprocessors = [PREPROCESSORS[name](**params) for name, params in preprocessors.items()]

    def __call__(self, text_batch: List[str]) -> List[str]:
        processed_batch = []
        for text in text_batch:
            for preprocessor in self.preprocessors:
                text = preprocessor(text)
            processed_batch.append(text)
        return processed_batch


PREPROCESSORS = {
    "base_cleaning": WhitespaceRemovePreprocessor,
    "lowcase": MakeLowerCasePreprocessor,
    "stem": StemPreprocessor,
    "lemma": LemmaPreprocessor,
}
