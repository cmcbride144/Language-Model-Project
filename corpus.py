import re
from nltk import word_tokenize


def tokenize(train_file):

    '''
    Function: tokenize()

        Description:
            Function should open and read a file line by line (or sentence by sentence), tokenize each line and append that line to a list
        Parameters:
            1 Parameter:
                A list of lists (nested list) of tokens
        Returns:
            List of lists (nested list) of tokenized tokens
                Output example:
                    Ex: [['why', ',', 'sir', ',', 'a', 'carpenter', '.'],...]
    '''

    tokens_list = []

    i = 0
    # Open and read the user_file line by line creating lists of lines in a larger list. Loop over these inner lists...
    for line in open(train_file).readlines():

        while i < 10:

            # If there is not an apostrophe (') in the line...
            if "'" not in line:

                # ...begin tokenization using NLTK word_tokenize(). Lowercase the first token in the line, don't lowercase following tokens.
                tokens = word_tokenize(line[0].lower() + line[1:])

                tokens_list.append(tokens)

                i += 1

    return tokens_list


def detokenize(tokens_list):

    '''
    Function: detokenize()

        Description:
            Function should join all tokens in a list to a single list of strings.
        Parameters:
            1 Parameter:
                A nested list of tokenized tokens
        Returns:
            List of detokenized strings, sentences.
                Output example:
                    Ex: ['Why, sir, a carpenter.']
    '''

    detokenized_text = []

    # Begin looping over tokenized sentences or lines
    for tokens in tokens_list:

        # Join tokens together as strings.
        tokens = ' '.join(tokens)

        # Removes all spaces before punctuation except apostrophe(').
        tokens = re.sub(r"\s([?.,:;!](?:\s|$))", r"\1", tokens)

        # Removes space before apostrophes(').
        tokens = re.sub(r"\s(\')", "\'", tokens)

        # Capitalizes tokens after start of sentence(^) or after punctuation[.?!].
        tokens = re.sub("(^|[.?!])\s*([a-zA-Z])", lambda p: p.group(0).upper(), tokens)


        detokenized_text.append(tokens)

    return detokenized_text
