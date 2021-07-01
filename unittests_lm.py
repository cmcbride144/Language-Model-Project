from lm import LanguageModel 
from lm import *
import unittest


class LMUnitTest(unittest.TestCase):
    
    '''
    Class: LMUnitTest
    
        Description:
            For class UnitTest, unittests_lm, bigrams (or LanguageModel(2)) were chosen for testing, 
            but methods should be valid for all ngrams.
        Methods:
            6 Methods: test_get_ngrams(), test_train(), test_normalize(), test_p_next(), 
                       test_sample(), test_generate(), test_beam_search()     
    '''
   

    def test_get_ngrams(self):
        
        ''' 
        LMUnitTest class Method: test_get_ngrams()
        
            Description:
                Tests get_ngrams() lm class Method.
                    Parameter: A nested list of tokens
                    Assertion: A nested list of user-specified ngram tuples
                Parameters:
                    No parameters.
                Returns:
                    No returns: 
                        Asserts that get_ngrams() returns a nested list of user-specified ngram tuples
        '''

        ngrams_test = [['Repent', 'kingdom'], ['heaven','at', 'hand']]
        
        ngrams_result = bigrams.get_ngrams(ngrams_test)
        
        self.assertEqual(ngrams_result, [[(None, 'Repent'), ('Repent', 'kingdom'), ('kingdom', None)], [(None, 'heaven'), ('heaven', 'at'), ('at', 'hand'), ('hand', None)]])


    def test_train(self):
        
        ''' 
        LMUnitTest class Method: test_train()
        
            Description:
                Tests train() lm class Method
                    Parameter: A nested list of tokens
                    Assertion: A dictionary of key, value pairs tuple history and key, value pairs token and frequency as prediction, respectivley
                Parameters:
                    No parameters.
                Returns:
                    No returns: 
                        Asserts that train()
        '''
        
        train_test = [['Repent', 'kingdom'], ['heaven','at', 'hand']]
        
        train_result = bigrams.train(train_test)
        
        self.assertEqual(train_result, {(None,): {'Repent': 1, 'heaven': 1}, ('Repent',): {'kingdom': 1}, ('kingdom',): {None: 1}, ('heaven',): {'at': 1}, ('at',): {'hand': 1}, ('hand',): {None: 1}})


    def test_normalize(self):
        
        ''' 
        LMUnitTest class Method: test_normaliz()
        
            Description:
                Tests normalize() lm class Method
                    Parameter: A dictionary of multiple key, value pairs of token and frequency of that token, respectively
                    Assertion: A dictionary of multiple key, value pairs of token and probability distribution of that token, respectively
                Parameters:
                    No parameters.
                Returns:
                    No returns: 
                        Asserts that normalize() returns multiple key, value pairs of tokens and probability distributions of those tokens, respectively
        '''
        
        normalize_test = {'Repent': 1, 'heaven': 1}
        
        normalize_result = bigrams.normalize(normalize_test)
        
        self.assertEqual(normalize_result, {'Repent': 0.2857142857142857, 'heaven': 0.2857142857142857})
        
        
    def test_p_next(self):
        
        ''' 
        LMUnitTest class Method: test_p_next()
        
            Description:
                Tests p_next() lm class Method. 
                    Parameter: A list of tokens. 
                    Assertion: A key, value pair of a token and probability distribution, respectively.
                Parameters:
                    No parameters.
                Returns:
                    No returns: 
                        Asserts that p_next() returns a key, value pair of a token and probability distribution
        '''
    
        p_next_test = ['Repent', 'kingdom', 'heaven', 'at', 'hand']
        
        p_next_result = bigrams.p_next(p_next_test)
        
        self.assertEqual(p_next_result, {'hand': 0.3333333333333333})
        
        
    def test_sample(self):
        
        ''' 
        LMUnitTest class Method: test_sample()
        
            Description:
                Tests sample() lm class Method. 
                    Parameter: A key, value pair of a token and probability distribution, respectively.
                    Assertion: A list of a single string
                Parameters:
                    No parameters.
                Returns:
                    No returns: 
                        Asserts that sample() returns the desired result: a list of a single string
        '''
        
        sample_test = {'hand': 0.3333333333333333}
        
        sample_result = bigrams.sample(sample_test)
        
        self.assertEqual(sample_result, ['hand'])
        
        
        
    def test_beam_search(self):
        
        '''
        LMUnitTest class Method: test_beam_search()
        
            Description:
                No LMUnitTest class test yet tested for LanguageModel class method, beam_search()
            Status:
                Under construction...
        '''
        
        pass


if __name__ == '__main__':
    
    unittest.main()