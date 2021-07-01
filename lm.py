from corpus import *
from random import *
from math import log


class LanguageModel:    
    
    ''' 
    Class: LanguageModel
    
        Description:
            LanguageModel class instatiates a LanguageModel of n-grams
        Methods:
            6 Methods: get_ngrams(), train(), normalize(), p_next(), sample(), generate(), beam_search()
    '''
    
    
    # Initialize constructor
    def __init__(self, n):          
        
        # User-specified integer (2 = bigram model, 3 = trigram model etc.)
        self.n = n                  
        
        # Nested dictionary 'counts' (outer_dictionary)
        self.counts = {}            
        
        # Vocabulary set()
        self.vocabulary = set()     
        
        # Inner dictionary of nested dictionary 'counts'
        self.word_counts = {}        
        
        # Distribution dictionary
        self.distribution = {}      
                
    
    def get_ngrams(self, tokens_list): 
        
        ''' 
        LM class Method: get_ngrams()
    
            Description:
                Function should pad a list of lists of tokens and return ngram tuples for tokens in the inner lists
            Parameters:
                1 Parameter: 
                    A list of lists (nested list) of tokens tokenized from corpus.py
            Returns:
                N-grams of tokens 
                    Output example: 
                        Ex: [...[(None, None, 'why'), (None, 'why', ','), ('why', ',', 'sir'), (',', 'sir', ','), ('sir', ',', 'cobble'), (',', 'cobble', 'you'), ('cobble', 'you', '.'), ('you', '.', None), ('.', None, None)]]
        ''' 
        
        ngrams = [] 
        
        # Start looping through the input nested list.
        for line in tokens_list:  
            
            # Pad [None] for the number of n-grams.
            pad_list = [None] * (self.n-1) + line + [None] * (self.n-1)  
            
            # Loops through n tuples and merges tokens, making n-grams. Returns a list of tuples.
            ngram_tuples = list(zip(*[pad_list[i:] for i in range(self.n)]))  
            
            ngrams.append(ngram_tuples) 
        
        return ngrams 


    def train(self, token_list):
        
        ''' 
        LM class Method: train()
        
            Description:
                Function should train a nested list of tokens and return a 'counts' nested dictionary 
                of values dictionaries of keys tokens and values frequencies of the tokens in the inner lists
            Parameters:
                1 Parameter: 
                    A list of lists (nested list) of tokens tokenized from corpus.py
            Returns:
                No returns (Except counts outer dictionary and word_counts inner dictionary for testing purposes)
        ''' 
        
        # Start looping through the input nested list.
        for line in token_list:  
            
            # Start looping through the tokens in the inner list. 
            for token in line: 
                
                # Adds tokens to vocabulary set
                self.vocabulary.add(token) 
                
                # Checks if the token is a key in the inner dictionary of dictionary 'counts', called 'word_counts'
                if token in self.word_counts: 
                    
                    # If token is a key in 'word_counts', add 1 to the value which is frequency of the token in the vocabulary
                    self.word_counts[token] += 1 
               
                else: 
                    
                    # Initialize 1 as the 'word_counts' value 
                    self.word_counts[token] = 1 
        
        # Call get_ngrams on the input and get ngrams, a list of lists of ngrams:
        ngrams = self.get_ngrams(token_list)  
        
        # Start looping through inner list of ngrams.
        for gram in ngrams:  
            
            # Start looping through tuples of ngrams
            for tup in gram: 
                
                # Set history to all tokens before the last token in the gram (which is an ngram).
                history = tuple(tup[:-1])  
                
                # Set prediction to last token in the gram (which is an ngram).
                prediction = tup[-1]  
                
                # Check if the history is not a key in 'counts' dictionary
                if history not in self.counts.keys(): 
                    
                    # If history not a key in 'counts' dictionary, add a dictionary value with the prediction as key and initialize frequency as 1.
                    self.counts[history] = {prediction: 1}  
                
                else: 
                    
                    # If the prediction is not a key in history...
                    if prediction not in self.counts[history].keys(): 
                        
                        # ...add it as the prediction for the history and initilaize frequency as 1
                        self.counts[history][prediction] = 1 
                    
                    else: 
                        
                        # Add 1 to the freqency of prediction for the history
                        self.counts[history][prediction] += 1 
       
        # This is only for testing purposes
        counts = self.counts 
        
        # This is only for testing purposes
        return counts 


    def normalize(self, word_counts):
        
        ''' 
        LM class Method: normalize()
        
            Description:
                Function should normalize inner value frequencies from return new word_counts dictionary with normalized inner values as probabilities
            Parameters:
                1 Parameter: 
                    A dictionary 'word_counts'
            Returns:
                A dictionary 'word_counts' of keys word annd values float predictions 
                    Output example: 
                        Ex: {...'saucy': 0.006622516556291391, 'fellow': 0.006622516556291391, 'cobble': 0.006622516556291391}
        '''
        
        # Makes a copy of word_counts dictionary (does not change original 'word_counts' dictionary)
        new_wordcounts = word_counts.copy() 
        
        # Sums the frequencies of the values in the word_counts dictionary copy
        sum_freq = sum(new_wordcounts.values()) 
        
        # Begin looping over the key, prediction and value, frequency in new_wordcounts
        for prediction, freq in new_wordcounts.items(): 
            
            # Normalizes and applied add 1 smoothing.
            # Frequency + smoothing (0.01) divided by the total frequency for the sentence + the length of the vocabulary set
            new_wordcounts[prediction] = (freq + 0.01) / (sum_freq + len(self.vocabulary))   
                                                                                         
        return new_wordcounts 
        
    
    def p_next(self, token_sequence): 
        
        ''' 
        LM class Method: p_next()
        
            Description:
                Return the estimated probability distribution for the next word that occurs after the token sequence tokens
            Parameters:
                1 Parameter: 
                    A single, arbitrarily long list of tokens
            Returns:
                A dictionary that represents a probability distribution. 
                The keys of the returned dict are words which might follow tokens, and the values are the probabilities for each word, as floats. 
                    Output example: 
                        Ex: {'indeed': 0.020618556701030927}
        '''
        
        # Check if the length of token_sequence is greater than or equal to n-1
        if len(token_sequence) >= self.n-1: 
            
            # Set tokens to the tokens before n-1
            tokens = token_sequence[-self.n-1:] 
        
        else: 
            
            # Pad the tokens with [None] pad times n-1 minus the length of the token sequence, and then ad the token sequence.
            tokens = [None] * ((self.n-1) - len(token_sequence)) + token_sequence 
        
        tokens = tuple(tokens) 
        
        # Check if the tokens tuple (an ngram) is a key in the 'counts' dictionary
        if tokens in self.counts: 
            
            # If so, set distribution as the normalized frequency of the ngram
            distribution = self.normalize(self.counts[tokens]) 
        
        else:
        
            # Set distribution to the normalized frequency of the random_word in counts
            distribution = {token: 1 / len(self.counts) for token in self.vocabulary}
        
        return distribution 
                 
    
    def sample(self, distribution):
        
        ''' 
        LM class Method: sample()
        
            Description:
                Return the key from the input dictionary, distribution
            Parameters:
                1 Parameter: 
                    A probability distribution dictionary whose values are numbers adding up to 1
            Returns:
                A probability distribution key from the input dictionary, chosen according to its probability
                    Output example: 
                        Ex: ['meanest']
        '''
         
        # Gets random word as a list from a list of the distribution keys and list of distribution values
        sample = choices(list(distribution.keys()), list(distribution.values())) 
        
        return sample 

    
    def generate(self):  
        
        ''' 
        LM class Method: generate()
        
            Description:
                Generates a random token sequence according to the underlying probability distribution
            Parameters:
                No parameters.
            Returns:
                A full generated text, as a list of tokens (strings)
                    Output example: 
                        Ex: ['A trade, sir, cobble you.']
        '''
                
        tokens = [] 
        
        # Pad list with [None] times the user-specified n
        tokens = [None] * (self.n) 
        
        # Gets sample words based on probability distribution of tokens and add to tokens list
        tokens.extend(self.sample(self.p_next(tokens))) 
        
        # While the last token is not a None...
        while tokens[-1] != None: 
            
            # Gets sample words based on the probability distribution of the history tokens (all before the prediction n-1)...
            # ...and add token to the tokens list
            tokens.extend(self.sample(self.p_next(tokens[-(self.n-1):]))) 
        
        # Detokenizes the tokens from n until n-1 and sets it to variable 'generated text'
        generated_text = detokenize([tokens[self.n:-1]]) 
        
        return generated_text

    
    def beam_search(self, k): 
    
        ''' 
        LM class Method: beam_search()
        
            Description:
                Generates a token sequence according to the underlying probability distribution 
                in k possible highest probabilities for each consecutive token, will prune back to k, and will return a generated sentence
                once all beams have predicted 'None'.
                
            Status:
                Under construction... 
        '''
        
        k_likely_candidates = []
        
        start_tokens = [None] * (self.n)
        
        def search_recursive(stack, results):
            
            if len(stack) == 0:
                
                return results
            
            else:
            
                sentence = stack.pop()
                
                # Gets the word of the last tuple
                last_word = sentence[-1][0] 
                
                k_candidates = self.p_next([last_word])
                
                for candidate in k_candidates:
                
                    if candidate == None:
                    
                        results.append(sentence + None)
                    
                    else:
                        
                        # The LanguageModel class method: beam_search() is still under construction
                        pass