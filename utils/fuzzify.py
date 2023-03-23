import random
import string


class Fuzzifier:
    def __init__(self):
        pass


    def random_insertions(self, text, frequency):
        """
        Randomly insert characters into a string
        """
        result = []
        for char in text:
            result.append(char)
            if random.random() < frequency:
                result.append(random.choice(string.ascii_letters))
        return ''.join(result)


    def random_deleteions(self, text, frequency):
        """
        Randomly delete characters from a string
        """
        result = []
        for char in text:
            if random.random() > frequency:
                result.append(char)
        return ''.join(result)


    def random_substitutions(self, text, frequency):
        """
        Randomly substitute characters in a string
        """
        result = []
        for char in text:
            if random.random() < frequency:
                result.append(random.choice(string.ascii_letters))
            else:
                result.append(char)
        return ''.join(result)


    def random_transpositions(self, text, frequency):
        """
        Randomly transpose two adjacent characters in a string
        """
        result = []
        for i in range(len(text) - 1):
            if random.random() < frequency:
                result.append(text[i + 1])
                result.append(text[i])
            else:
                result.append(text[i])
        if len(text) > 0:
            result.append(text[-1])
        return ''.join(result)


    def fuzzify(self, text, corrupt_frequency=0.10):
        """
        Apply all fuzzification methods to a string
        """
        avg_frequency = corrupt_frequency / 4
        text = self.random_insertions(text, avg_frequency)
        text = self.random_deleteions(text, avg_frequency)
        text = self.random_substitutions(text, avg_frequency)
        text = self.random_transpositions(text, avg_frequency)
        return text
