from codinglab import (
    PrefixEncoderDecoder,
    PrefixCodeTree,
    SourceChar,
)
from collections import OrderedDict
from enum import Enum
from typing import Dict
import math


class BinaryAlphabet(str, Enum):
    zero = "0"
    one = "1"


class ShannonFanoEliasBinaryCoder(PrefixEncoderDecoder[SourceChar, BinaryAlphabet]):
    """
    Shannon-Fano-Elias Coder for prefix codes.

    This coder implements the Shannon-Fano-Elias coding algorithm, which
    constructs prefix codes based on the cumulative distribution function.
    For each symbol x with probability p(x), the code length is:
    l(x) = -⌈log₂(p(x))⌉ + 1
    The code is the binary representation of the modified cumulative
    probability F̄(x) = Σ_{a<x} p(a) + p(x)/2, truncated to l(x) bits.

    Attributes:
        _probabilities: Ordered dictionary mapping source symbols to their probabilities
        _cumulative_probs: Dictionary mapping symbols to their cumulative probabilities
        _modified_cumulative: Dictionary mapping symbols to their modified cumulative F̄(x)
    """

    def __init__(
        self,
        probabilities: OrderedDict[SourceChar, float],
    ) -> None:
        """
        Initialize the Shannon-Fano-Elias encoder with symbol probabilities.

        Args:
            probabilities: Ordered dictionary mapping source symbols to their
                           probabilities (must sum to 1.0). The order determines
                           the cumulative probability calculation.

        Raises:
            ValueError: If probabilities don't sum to approximately 1.0
                       or if any probability is non-positive
        """
        prob_sum = sum(probabilities.values())
        if not math.isclose(prob_sum, 1.0, rel_tol=1e-9):
            raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum}")

        for symbol, prob in probabilities.items():
            if prob <= 0:
                raise ValueError(
                    f"Probability for symbol '{symbol}' must be positive, got {prob}"
                )

        self._probabilities = probabilities
        self._channel_alphabet = [BinaryAlphabet.zero, BinaryAlphabet.one]
        super().__init__(list(probabilities.keys()), self._channel_alphabet)

    def _build_prefix_code_tree(self) -> None:
        cumulative = 0.0
        self._tree = PrefixCodeTree()

        for symbol, prob in self._probabilities.items():
            f = cumulative + prob / 2.0
            code_length = math.ceil(-math.log2(prob)) + 1

            binary = []
            frac = f
            for _ in range(code_length):
                frac *= 2
                if frac >= 1:
                    binary.append(BinaryAlphabet.one)
                    frac -= 1
                else:
                    binary.append(BinaryAlphabet.zero)

            self._tree.insert_code(binary, symbol)
            cumulative += prob

        self._build_table_from_tree()

    @property
    def codes(self) -> Dict[SourceChar, str]:
        if hasattr(self, "_code_table") and self._code_table:
            return {k: "".join(v) for k, v in self._code_table.items()}
        return {}

    @property
    def expected_code_length(self) -> float:
        """
        Calculate the expected code length.

        Returns:
            Expected number of channel symbols per source symbol,
            weighted by symbol probabilities
        """
        if not self._code_table:
            return 0.0

        total = 0.0
        for symbol, prob in self._probabilities.items():
            if symbol in self._code_table:
                total += prob * len(self._code_table[symbol])
        return total

    @property
    def entropy(self) -> float:
        """
        Calculate the Shannon entropy of the source.

        Returns:
            Shannon entropy in bits (for binary channel)
        """
        h = 0.0
        for prob in self._probabilities.values():
            if prob > 0:
                h -= prob * math.log2(prob)
        return h

    @property
    def coding_efficiency(self) -> float:
        """
        Calculate the coding efficiency.

        Returns:
            Ratio of entropy to expected code length,
            representing how close the code is to optimal
        """
        expected_len = self.expected_code_length
        if expected_len == 0:
            return 0.0
        return self.entropy / expected_len
