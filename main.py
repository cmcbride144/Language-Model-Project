from corpus import *
from lm import LanguageModel
import sys


while True:

    print("\nWelcome to Language Model Studio.\n"
          "\nThis is a simple application where you can create a new language model of your specification.\n"
          "You will be able to load text files and train your own language model on those texts.\n"
          "You can then print the generated text from your language model to your screen and also write multiple texts to files.\n"
          "You can exit the program at any time by typing 'ctrl + c'.\n")

    print("\nPlease note that all of your files must be located in the same folder as main.py.")

    print("\nPlease type in your files' name and then press 'enter'.")

    try:
        user_file = input()

        train_file = user_file
        train_file = tokenize(train_file)

        print("\nFile successfully uploaded.")

    except:

        print("That is an invalid file name. Please type a valid file name or type 'exit' to exit the program.")

        continue


    print("\nPlease specify which n-gram you would like to train as an integer and press 'enter'.")

    try:
        user_n = int(input())
    except:
        print("\nPlease input an integer.")

        continue

    lm = LanguageModel(user_n)


    print("\nOk. You specified to train a {}-gram Language Model.".format(user_n))

    print("\nReady to start training. To start training, please press 'enter'.")

    start_training = input()

    print('\nStarting training for a {}-gram Language Model on {} ...'.format(user_n, user_file))

    lm.train(train_file)

    print("\nAll done training! You can now generate texts.")

    print("\nYou have the option of generating a text with the regular {}-gram Language Model or with Beam Search."
          "\nPlease type 'beam' if you would like to use Beam Search, otherwise press 'enter' for default generation.".format(user_n))


    generate_method = input()

    if generate_method == 'beam':

        print("\nWe're sorry: Beam Search has yet to be implemented and is under construction. Please generate using the default generator. We're sorry for the inconvenience.")

        print("\nHow large would you like your beam to be? Please enter an integer and then press 'enter'.")
        user_beam_size = int(input())

        print("\nPlease type as an integer how many texts you would like to generate. Then press 'enter'.")
        user_beam_text_num = int(input())

        print("\nOk. You have selected a Beam Search with beam size {}. Starting generation with beam size of {}...".format(user_beam_size))
        lm.beam_search(user_beam_size)

        print("\nGenerating {} text(s) with Beam Search...".format(user_beam_text_num))
        print("\nHere are your {}-gram text(s):".format(user_n))

        user_beam_list = []
        for num in range(0, int(user_beam_text_num)):
            user_beam_generated_text = lm.generate()
            user_beam_list += user_beam_generated_text

            print(user_beam_generated_text)

    else:
        print("\nPlease type as an integer how many texts you would like to generate. Then press 'enter'.")
        user_text_num = int(input())

        print("\nGenerating {} text(s)...".format(user_text_num))
        print("\nHere are your {}-gram text(s):".format(user_n))

        user_list = []
        for num in range(0, int(user_text_num)):
            user_generated_text = lm.generate()
            user_list += user_generated_text

            print(user_generated_text)

    print("\nYou can now download your generated text file! Please type in the desired name for your file as 'filename'.txt and press 'enter'.")

    new_file_name = input()


    user_generated_file = open(new_file_name, 'a')
    for line in user_generated_text:
        print(line)
        user_generated_file.write(line + '\n')
    user_generated_file.close()


    print("\nYour file has been successfully downloaded to this folder.")

    print("\nIf you are finished with Language Model Studio, please type 'ctrl + c'.\n"
          "If you would like to upload a new file, train and generate another text, please type 'train' and press 'enter'.")


    train_again = input()

    if train_again == 'train':

        print('\nHeading back to start...')

    else:

        print('\nExiting Language Model Studio...')

        sys.exit()
