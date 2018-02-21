#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2018
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
######################################################################
import csv
import math

import numpy as np

from movielens import ratings
from random import randint

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, is_turbo=False):
      self.name = 'flixbot'
      self.is_turbo = is_turbo
      self.read_data()
      self.user_vector = [0.] * 9125
      self.data_points = 0

    def greeting(self):
      """chatbot greeting message"""
      greeting_message = 'Hi! I\'m FlixBot! I\'m going to recommend a movie to you. ' \
                         'First I will ask you about your taste in movies. ' \
                         'Tell me about a movie that you have seen.'

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      goodbye_message = 'Thank you for hanging out with me! Stay in touch! Goodbye!'

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################

    def get_movie_and_sentiment(self, input):
        if input.count('\"') != 2:
            return None, None

        first_quote = input.find('\"') + 1
        second_quote = first_quote + input[first_quote:].find('\"')
        movie = input[first_quote:second_quote]

        # TODO: sentiment analysis
        sentiment = 5.0

        return movie, sentiment

    def update_user_vector(self, movie, sentiment):
        found_title = False
        i = 0
        for i, title in enumerate(self.titles):
            if title[0] == movie:
                found_title = True
                break

        if found_title:
            if self.user_vector[i] == 0.0:
                self.data_points += 1
            self.user_vector[i] = sentiment

    def process(self, input):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method, possibly#
      # calling other functions. Although modular code is not graded, it is       #
      # highly recommended                                                        #
      #############################################################################
      if self.is_turbo == True:
        response = 'processed %s in creative mode!!' % input
      else:
          movie, sentiment = self.get_movie_and_sentiment(input)
          if not movie:
              response = 'Sorry, I don\'t understand. Tell me about a movie that you have seen.'
          else:
              self.update_user_vector(movie, sentiment)

              response = 'Glad to hear you liked \"%s\"! ' if sentiment >= 2.5 else 'Sorry you didn\'t like \"%s\". '
              response = response % movie
              if self.data_points < 5:
                  response += 'Tell me about another movie you have seen.'
              else:
                  response += 'That\'s enough for me to make a recommendation.'
                  # TODO: make recommendation

      return response


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)
      print(len(self.titles), self.titles)
      print(len(self.ratings), self.ratings)
      print(self.sentiment)


    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""

      pass


    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure

      pass


    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot

      pass


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
      """Returns debug information as a string for the input string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
      return self.name


if __name__ == '__main__':
    Chatbot()
