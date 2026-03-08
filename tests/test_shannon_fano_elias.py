"""Tests for shannon_fano_elias encoder"""

import pytest
import math
from collections import OrderedDict
from codinglab.shannon_fano_elias import ShannonFanoEliasBinaryCoder


class TestShannonFanoEliasBinaryCoder:
    def test_valid_probabilities(self):
        """Создание с правильными вероятностями"""
        probs = OrderedDict({"a": 0.5, "b": 0.5})
        coder = ShannonFanoEliasBinaryCoder(probs)
        assert coder is not None

    def test_invalid_sum_raises(self):
        """Сумма вероятностей != 1.0 -- ошибка"""
        probs = OrderedDict({"a": 0.5, "b": 0.3})
        with pytest.raises(ValueError):
            ShannonFanoEliasBinaryCoder(probs)

    def test_negative_probability_raises(self):
        """Отрицательная вероятность -- ошибка"""
        probs = OrderedDict({"a": 1.5, "b": -0.5})
        with pytest.raises(ValueError):
            ShannonFanoEliasBinaryCoder(probs)

    def test_symbols(self):
        """Коды сгенерированы для всех символов"""
        probs = OrderedDict({"a": 0.75, "b": 0.025, "c": 0.1, "d": 0.05, "e": 0.075})
        coder = ShannonFanoEliasBinaryCoder(probs)
        assert len(coder.codes) == 5
        assert set(coder.codes.keys()) == {"a", "b", "c", "d", "e"}

    def test_binary(self):
        """Все коды из 0 и 1"""
        probs = OrderedDict({"a": 0.5, "b": 0.5})
        coder = ShannonFanoEliasBinaryCoder(probs)
        for code in coder.codes.values():
            assert all(c in "01" for c in code)

    def test_prefix_free(self):
        """Ни один код не префикс другого"""
        probs = OrderedDict({"a": 0.75, "b": 0.025, "c": 0.1, "d": 0.05, "e": 0.075})
        coder = ShannonFanoEliasBinaryCoder(probs)
        codes = list(coder.codes.values())
        for i, c1 in enumerate(codes):
            for j, c2 in enumerate(codes):
                if i != j:
                    assert not c2.startswith(c1)

    def test_code_length(self):
        """Длина кода = ⌈-log₂(p)⌉ + 1"""
        probs = OrderedDict({"a": 0.75, "b": 0.25})
        coder = ShannonFanoEliasBinaryCoder(probs)
        assert len(coder.codes["a"]) == math.ceil(-math.log2(0.75)) + 1
        assert len(coder.codes["b"]) == math.ceil(-math.log2(0.25)) + 1

    def test_encode_decode(self):
        """Кодирование и декодирование обратно приводят к изначальному результату"""
        probs = OrderedDict({"a": 0.75, "b": 0.025, "c": 0.1, "d": 0.05, "e": 0.075})
        coder = ShannonFanoEliasBinaryCoder(probs)
        message = "abcde"
        encoded = coder.encode(message)
        decoded = coder.decode(encoded)
        assert "".join(decoded) == message

    def test_entropy(self):
        """Энтропия H = -Σ p·log₂(p)"""
        probs = OrderedDict({"a": 0.75, "b": 0.25})
        coder = ShannonFanoEliasBinaryCoder(probs)
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        assert abs(coder.entropy - expected) < 1e-6

    def test_expected_length(self):
        """Длина L = Σ p·l(x)"""
        probs = OrderedDict({"a": 0.75, "b": 0.25})
        coder = ShannonFanoEliasBinaryCoder(probs)
        expected = 0.75 * len(coder.codes["a"]) + 0.25 * len(coder.codes["b"])
        assert abs(coder.expected_code_length - expected) < 1e-6

    def test_efficiency(self):
        """Эффективность <= 1"""
        probs = OrderedDict({"a": 0.75, "b": 0.025, "c": 0.1, "d": 0.05, "e": 0.075})
        coder = ShannonFanoEliasBinaryCoder(probs)
        assert 0 < coder.coding_efficiency <= 1

    """тесты для эксперимента"""

    def test_task1_probabilities(self):
        """Вероятности"""
        probs = OrderedDict({"a": 0.75, "b": 0.025, "c": 0.1, "d": 0.05, "e": 0.075})
        coder = ShannonFanoEliasBinaryCoder(probs)
        assert len(coder.codes) == 5

    def test_task1_encode_1000_symbols(self):
        """Кодирование 1000 символов"""
        import random

        random.seed(42)
        probs = OrderedDict({"a": 0.75, "b": 0.025, "c": 0.1, "d": 0.05, "e": 0.075})
        coder = ShannonFanoEliasBinaryCoder(probs)
        symbols = list(probs.keys())
        weights = list(probs.values())
        message = "".join(random.choices(symbols, weights=weights, k=1000))
        encoded = coder.encode(message)
        decoded = coder.decode(encoded)
        assert "".join(decoded) == message
        assert len(encoded) > 0
