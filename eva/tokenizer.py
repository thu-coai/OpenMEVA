"""A module for Tokenizer"""
import re

from nltk.tokenize import WordPunctTokenizer
import spacy

class SimpleTokenizer():
	'''
	A simple tokenizer. ``method`` can either be ``nltk``or ``space`` or ``spacy'' or ``spacy_zh''.
	If ``nltk``, use ``WordPunctTokenizer`` from ``nltk.tokenize``.
	If ``space``, use ``str.split(" ")``.
	If ``spacy``, use ``spacy.load("en_core_web_sm")``.
	If ``spacy_zh``, use ``spacy.load("zh_core_web_sm")``.

	Arguments:
		method (str): the tokenization method, ``nltk`` or ``space``.
		special_tokens (List[str]): special tokens not to tokenize, such as ``<go>``.
	'''
	def __init__(self, method, special_tokens = None):
		self.method = method
		self.special_tokens = special_tokens

		if method == "nltk":
			self._callable_tokenizer = WordPunctTokenizer().tokenize
		elif method == "space":
			self._callable_tokenizer = str.split
		elif method == "spacy":
			# python -m spacy download en_core_web_sm
			self._callable_tokenizer = spacy.load('en_core_web_sm')
		elif method == "spacy_zh":
			# python -m spacy download zh_core_web_sm
			self._callable_tokenizer = spacy.load('zh_core_web_sm')
		else:
			raise ValueError('`method` is invalid value {}, should be "nltk" or "space" or "spacy" '.format(method))

	def tokenize(self, sentence):
		'''Tokenize a sentence to a list of tokens.

		Arguments:
			sentence (str): a sentence to tokenize.
		'''
		if self.special_tokens is None:
			return self._callable_tokenizer(sentence)
		regexPattern = '(' + '|'.join(map(re.escape, self.special_tokens)) + ')'
		segments = re.split(regexPattern, sentence)
		sent = []
		for seg in segments:
			if seg not in self.special_tokens:
				sent += self._callable_tokenizer(seg.strip())
			else:
				sent += [seg]
		return sent

	def convert_tokens_to_sentence(self, tokens):
		'''Convert tokens to sentence.
		It usually works like the reverse operation of :meth:`tokenize`, but it is not gauranteed.
		It may like ``" ".join(tokens)``, but some special condition and tokens will be took care.

		Arguments:
			tokens(List[str]): tokenized sentence
		'''		
		if self.method == "nltk":
			sent = " ".join(tokens)
			out_string = sent.replace(' .', '.').replace(' ?', '?'). \
				replace(' !', '!').replace(' ,', ',').replace(" ' ", "'"). \
				replace(" n't", "n't").replace(" 'm", "'m"). \
				replace(" 's", "'s"). \
				replace(" 've", "'ve").replace(" 're", "'re")
			return out_string
		elif self.method == "space" or self.method == "spacy" or self.method == "spacy_zh":
			return " ".join(tokens)
		else:
			raise RuntimeError("No such tokenizer %s" % self.method)
	
	def name(self):
		return "SimpleTokenizer/" + self.method 

class PretrainedTokenizer():
	'''Bases: :class:`.dataloader.Tokenizer`

	A wrapper for ``Pretrainedtokenizer`` from ``transformers`` package.
	If you don't want to do tokenization on some special tokens, see
	``transformers.Pretrainedtokenizer.add_special_tokens``.

	Arguments:
		tokenizer (transformers.Pretrainedtokenizer): An
			instance of ``transformers.Pretrainedtokenizer``.
	'''
	def __init__(self, method):
		if "gpt" in method:
			from transformers.tokenization_gpt2 import GPT2Tokenizer
			self.tokenizer = GPT2Tokenizer.from_pretrained(method)
		elif "bert" in method:
			from transformers.tokenization_bert import BertTokenizer
			self.tokenizer = BertTokenizer.from_pretrained(method)
		else:
			raise ValueError('`method` is invalid value {}, should be "gpt"/"bpe" or "bert"'.format(method))

		self._tokenizer_class_name = self.tokenizer.__class__.__name__

	def tokenize(self, sentence):
		return self.tokenizer.tokenize(sentence)

	def convert_tokens_to_sentence(self, tokens):
		return self.tokenizer.convert_tokens_to_string(tokens)

	def name(self):
		return "PretrainedTokenizer/" + self.method 