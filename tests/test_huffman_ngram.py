"""Tests for Huffman n-gram coder."""

from codinglab.huffman_ngram import HuffmanNGramCoder


class TestHuffmanNGramCoder:
    def test_ngram_extraction(self):
        """Тест: разбиение на n-граммы работает"""
        coder = HuffmanNGramCoder(n=2)
        text = "abcdef"
        ngrams = coder._extract_ngrams(text)
        assert ngrams == ["ab", "cd", "ef"]

    def test_padding(self):
        """Тест: дополнение до длины кратной n"""
        coder = HuffmanNGramCoder(n=3)
        text = "abcde"
        padded = coder._pad_text(text)
        assert len(padded) == 6
        assert padded.endswith(coder._padding_symbol)

    def test_fit(self):
        """Тест: fit() строит таблицу кодов"""
        coder = HuffmanNGramCoder(n=2)
        text = "aabbab"
        coder.fit(text)
        assert len(coder._code_table) > 0
        assert len(coder._frequencies) > 0

    def test_encode_decode(self):
        """Тест: кодирование и декодирование обратно приводят к изначальному результату"""
        coder = HuffmanNGramCoder(n=2)
        text = "aabbab"
        coder.fit(text)
        encoded = coder.encode(text)
        decoded = coder.decode(encoded)
        assert decoded == text

    def test_entropy_per_symbol(self):
        """Тест: энтропия на символ считается"""
        coder = HuffmanNGramCoder(n=2)
        text = "aabbab"
        coder.fit(text)
        assert coder.entropy_per_symbol > 0
        assert coder.entropy > 0

    def test_expected_length_per_symbol(self):
        """Тест: средняя длина на символ считается"""
        coder = HuffmanNGramCoder(n=2)
        text = "aabbab"
        coder.fit(text)
        assert coder.expected_code_length_per_symbol > 0

    def test_efficiency(self):
        """Тест: эффективность в диапазоне (0, 1]"""
        coder = HuffmanNGramCoder(n=2)
        text = "aabbab"
        coder.fit(text)
        assert 0 < coder.coding_efficiency <= 1

    def test_different_n_different_efficiency(self):
        """Тест: разные n дают разную эффективность"""
        text = "aabbab" * 100

        coder1 = HuffmanNGramCoder(n=1)
        coder1.fit(text)

        coder2 = HuffmanNGramCoder(n=2)
        coder2.fit(text)

        assert (
            coder1.coding_efficiency != coder2.coding_efficiency
            or coder1.expected_code_length_per_symbol
            != coder2.expected_code_length_per_symbol
        )
