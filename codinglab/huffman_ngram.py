"""Huffman coder for n-grams."""

import math
import heapq
from collections import Counter
from typing import Dict, List, Optional, Sequence

from codinglab import PrefixEncoderDecoder, PrefixCodeTree, SourceChar, ChannelChar
from codinglab.tree import TreeNode
from enum import Enum


class BinaryAlphabet(str, Enum):
    """Binary alphabet for channel symbols."""

    zero = "0"
    one = "1"


class HuffmanNode(TreeNode[ChannelChar, SourceChar]):
    """Node in the Huffman tree during construction."""

    def __init__(
        self,
        freq: float = 0.0,
        value: Optional[SourceChar] = None,
        children: Optional[Dict[str, "HuffmanNode"]] = None,
    ) -> None:
        super().__init__(value=value, children=children)
        self.freq = freq

    def __lt__(self, other: "HuffmanNode") -> bool:
        return self.freq < other.freq


class HuffmanNGramCoder(PrefixEncoderDecoder[str, BinaryAlphabet]):
    """
    Huffman coder for n-grams.

    This coder implements Huffman coding for blocks of n symbols (n-grams).
    It builds a Huffman tree based on frequencies of n-grams in the text.
    """

    def __init__(self, n: int = 1, padding_symbol: str = "\x00") -> None:
        """
        Initialize the Huffman n-gram coder.

        Args:
            n: Length of n-grams (default: 1 for standard Huffman)
            padding_symbol: Symbol used for padding (default: null character)
        """
        self.n = n
        self._padding_symbol = padding_symbol
        self._code_table: Optional[Dict[str, Sequence[BinaryAlphabet]]] = None
        self._reverse_table: Dict[str, str] = {}
        self._frequencies: Dict[str, float] = {}
        self._tree: Optional[PrefixCodeTree] = None

        self._source_alphabet: List[str] = []
        self._channel_alphabet: List[BinaryAlphabet] = [
            BinaryAlphabet.zero,
            BinaryAlphabet.one,
        ]

        # Вызываем родительский инициализатор с пустыми алфавитами
        # Они будут обновлены в fit()
        super().__init__(self._source_alphabet, self._channel_alphabet)

    def fit(self, text: str) -> None:
        """
        Build Huffman code table from text.

        Args:
            text: Training text to learn n-gram frequencies
        """
        ngrams = self._extract_ngrams(text)

        counter = Counter(ngrams)
        total = sum(counter.values())
        self._frequencies = {ngram: count / total for ngram, count in counter.items()}

        self._source_alphabet = list(self._frequencies.keys())

        # Обновляем алфавиты в родительском классе
        # (если базовый класс требует)

        self._build_prefix_code_tree()

    def _extract_ngrams(self, text: str) -> List[str]:
        """Extract n-grams from text (with padding if needed)."""
        padded_text = self._pad_text(text)

        ngrams: List[str] = []
        for i in range(0, len(padded_text), self.n):
            ngram = padded_text[i : i + self.n]
            if len(ngram) == self.n:
                ngrams.append(ngram)

        return ngrams

    def _pad_text(self, text: str) -> str:
        """Pad text to length divisible by n."""
        remainder = len(text) % self.n
        if remainder == 0:
            return text
        padding_needed = self.n - remainder
        return text + (self._padding_symbol * padding_needed)

    def _build_prefix_code_tree(self) -> None:
        """Build Huffman tree using the priority queue algorithm."""
        if not self._frequencies:
            return

        heap: List[HuffmanNode] = []
        for ngram, freq in self._frequencies.items():
            heapq.heappush(heap, HuffmanNode(freq=freq, value=ngram))

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent_node: HuffmanNode = HuffmanNode(
                freq=left.freq + right.freq,
                value=None,
                children={"0": left, "1": right},
            )
            heapq.heappush(heap, parent_node)

        root_huffman = heap[0] if heap else None
        self._tree = PrefixCodeTree(root_huffman)
        self._build_table_from_tree()

    def encode(self, text: Sequence[str]) -> Sequence[BinaryAlphabet]:
        """Encode text using n-gram Huffman codes."""
        if not self._code_table:
            raise ValueError("Code table not built. Call fit() first.")

        # Конвертируем входные данные в строку
        if isinstance(text, (list, tuple)):
            text_str = "".join(text)
        else:
            text_str = str(text)

        self._pad_text(text_str)
        ngrams = self._extract_ngrams(text_str)

        encoded_parts: List[BinaryAlphabet] = []
        for ngram in ngrams:
            if ngram in self._code_table:
                encoded_parts.extend(self._code_table[ngram])
            else:
                raise ValueError(f"N-gram '{ngram}' not found in code table")

        return encoded_parts

    def decode(self, encoded: Sequence[BinaryAlphabet]) -> Sequence[str]:
        if not self._tree:
            raise ValueError("Tree not built. Call fit() first.")

        # ✅ Передаём encoded напрямую, без конвертации
        decoded_ngrams = super().decode(encoded)

        decoded_text = "".join(decoded_ngrams)
        if decoded_text.endswith(self._padding_symbol):
            decoded_text = decoded_text.rstrip(self._padding_symbol)

        return [decoded_text]

    @property
    def expected_code_length(self) -> float:
        """Calculate expected code length per n-gram."""
        if not self._code_table:
            return 0.0

        total = 0.0
        for ngram, freq in self._frequencies.items():
            if ngram in self._code_table:
                total += freq * len(self._code_table[ngram])
        return total

    @property
    def expected_code_length_per_symbol(self) -> float:
        """Calculate expected code length per original symbol."""
        return self.expected_code_length / self.n if self.n > 0 else 0.0

    @property
    def entropy(self) -> float:
        """Calculate Shannon entropy of n-gram distribution."""
        h = 0.0
        for freq in self._frequencies.values():
            if freq > 0:
                h -= freq * math.log2(freq)
        return h

    @property
    def entropy_per_symbol(self) -> float:
        """Calculate Shannon entropy per original symbol."""
        return self.entropy / self.n if self.n > 0 else 0.0

    @property
    def coding_efficiency(self) -> float:
        """Calculate coding efficiency."""
        expected_len = self.expected_code_length
        if expected_len == 0:
            return 0.0
        return self.entropy / expected_len

    @property
    def redundancy(self) -> float:
        """Calculate redundancy."""
        return self.expected_code_length - self.entropy

    @property
    def redundancy_per_symbol(self) -> float:
        """Calculate redundancy per original symbol."""
        return self.redundancy / self.n if self.n > 0 else 0.0

    @property
    def alphabet_size(self) -> int:
        """Get size of n-gram alphabet."""
        return len(self._frequencies)
