#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2018
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
# Koooooby Shoob Kaylie
######################################################################
import csv
import math
import random
from PorterStemmer import PorterStemmer as ps
import re

import numpy as np

from movielens import ratings
from random import randint

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, is_turbo=False):
      self.prevRec = set([])
      self.name = 'flixbot'
      self.p = ps()
      self.threshold = 3.0
      self.is_turbo = is_turbo
      self.read_data()
      self.binarize()
      self.user_vector = [0.] * 9125
      self.data_points = 0
      self.recommendingMovies = False
      self.numRecs = 0
      
      self.negateSet= set([ "ain't", 'aversion', "can't", 'cannot', 'contradictory', 'contrary', 'counteract', 'dispute', 'dispute', "didn't", "don't", 'implausibility', 'impossibility', 'improbability', 'inability', 'incapable', 'incomplete', 'insignificant', 'insufficient', 'negate', 'negation', 'neither', 'never', 'no', 'no', 'no', 'no', 'nobody',  'non', 'none', 'nor', 'not', 'nothing', 'opposite', 'rather', 'unsatisfactory' 'untrue', "won't"])


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

    def is_a_movie(self, title):
        if title in self.titleDict.keys():
            return title + " (" + self.titleDict[title][0] + ")"
        return None

    ##returns title and input(with title extracted)
    ##     example: I like Titanic a lot -> Titanic (1997), I like a lot
    ##returns first moive, so consider recalling it w just input
    ##for disambiguate, consider returning flag for input since input is not needed
    ##     example
    def extract_movie(self, input):
        if input.count('\"') == 2:
            first_quote = input.find('\"') + 1
            second_quote = first_quote + input[first_quote:].find('\"')
            movie = input[first_quote:second_quote]
            input = input[:first_quote-2] + input[second_quote+1:]
            return movie, input
        pat = re.compile('([A-Z])')
        for m in pat.finditer(input):
            titleTest = input[m.start():]
            titleTest += ' '
            while " " in titleTest:
                titleTest = titleTest.rsplit(' ', 1)[0]
                firstWord = titleTest.split()[0]
                if firstWord.lower() == "an" or firstWord.lower() == "the" or firstWord.lower() == "a":
                    titleTest = titleTest + ', ' + firstWord
                    titleTest = titleTest.split(' ', 1)[1]  #after article handling, if needed
                    print titleTest
                fullTitle = self.is_a_movie(titleTest)
                if fullTitle:
                    if m.start() + len(titleTest) >= len(input):
                        return fullTitle, input[:m.start()-1]
                    else:
                        return fullTitle, input[:m.start()-1] + input[m.start()+len(titleTest)]
        return None, None
                
        
    def get_movie_and_sentiment(self, input):
        movie, input = self.extract_movie(input)
        if not movie and not input:
            return None, None, None

        orig_movie = movie #before article handling, readable version

        firstWord = movie.split()[0]
        lastWordInd = movie.index(movie.split()[-1])
        if firstWord.lower() == "an" or firstWord.lower() == "the" or firstWord.lower() == "a":
            movie = movie[:lastWordInd-1] + ', ' + firstWord + " " + movie[lastWordInd:]
            movie = movie.split(' ', 1)[1]  #after article handling, if needed

        if movie not in self.titleSet:
            return "NO_TITLE", "NO_TITLE", 0.0
        
        pos_neg_count, sentiment = 0, 0.0
        inv = 1
        for word in input.split(' '):
            word = self.p.stem(word)
            if word in self.negateSet:
                inv *= -1
            if word in self.sentiment:
                if self.sentiment[word] == 'pos':
                    pos_neg_count += 1 * inv
                else:
                    pos_neg_count -= 1 * inv
        if pos_neg_count > 0.0:
            sentiment = 1.0
        elif pos_neg_count < 0.0:
            sentiment = -1.0
        else:
            sentiment = 0.0
        return orig_movie, movie, sentiment

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

    #gets random response based on sentiment and movie
    def getResponse(self, sentiment):
      neutralSet = ['I wasn\'t quite sure if you liked \"%s\"...could you phrase that differently? ', 'So did you like \"%s\" or not? ', 'What\'s your opinion on \"%s\"? ', 'You seem to have mixed feelings about \"%s\". Do you mind elaborating? ', 'I can\'t tell if you liked \"%s\". Could you elaborate? ']
      posSet = ['Glad to hear you liked \"%s\"! ', 'Yea, \"%s\" was a great movie! ', 'I agree, \"%s\" was a modern masterpiece! ', 'Same, \"%s\" was an instant classic! ', 'Yessss, \"%s\" was life changing! ', 'Yea, I\'d give \"%s\" at least a 10/10, it was great! ', 'You have THE BEST opinions on movies, \"%s\" was great! ', 'I feel you,  \"%s\" was just incredible. ', 'Oh yea, \"%s\" was the best movie ever made. ']
      negSet = ['Sorry you didn\'t like \"%s\". ', 'Yea, I didn\'t like  \"%s\" either. ', 'Definitely, \"%s\" was just a bad experience. I was dragged along. ', 'Yea...\"%s\" was the worst movie I ever saw. ', 'I agree, \"%s\" made me cry in a bad way. ', 'I feel you,  \"%s\" was just bad. ', 'You should be a movie critic,  \"%s\" was objectively bad. ']
      if sentiment == 0.0:
          response = random.sample(neutralSet, 1)[0]
      else:
        if sentiment == 1.0:
            response = random.sample(posSet, 1)[0]
        else:
            response = random.sample(negSet, 1)[0]
      return response

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
      response = ''
      if self.is_turbo == True:
        response = 'processed %s in creative mode!!' % input
      else:
          orig_movie, movie, sentiment = self.get_movie_and_sentiment(input)
          maybe = False
          if self.data_points >= 5:
              if input.lower() == 'no' or self.numRecs >= 5:
                  #keeping previous sentiment
                  self.data_points = 0
                  self.numRecs = 0
                  response = "Okay! Tell me more about movies!"
                  return response
              elif input.lower() != 'yes':
                  response += "I'm not sure what you said! Please answer \'yes\' or \'no\'"
                  maybe = True
          if self.data_points < 5:
              if not movie:
                  response = 'Sorry, I don\'t understand. Tell me about a movie that you have seen.'
              elif movie == 'NO_TITLE':
                  response = 'Sorry, I\'m not familiar with that title.'
              elif sentiment == 0.0:
                  response = self.getResponse(sentiment) % movie
              else:
                  self.update_user_vector(movie, sentiment) # uses article-handled "X, The" version for title recognition
                  # response = 'Glad to hear you liked \"%s\"! ' if sentiment == 1.0 else 'Sorry you didn\'t like \"%s\". '
                  response = self.getResponse(sentiment) % orig_movie #uses human-readable, non-article-handled "The X" version for readability
                  if self.data_points < 5:
                      response += 'Tell me about another movie you have seen.'
          if self.data_points >= 5 and not maybe:
              response += '\nThat\'s enough for me to make a recommendation.'
              currRec = self.recommend(self.user_vector)
              for rec in currRec:
                  if rec not in self.prevRec:
                      response += '\nYou should watch ' + rec + '.'
                      self.prevRec.add(rec)
                      self.numRecs += 1
                      break
              response += '\nWould you like to hear another recommendation? (Or enter :quit if you\'re done.)'
              

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
      self.titleSet = set(item[0] for item in self.titles) #don't want to deal with binary search
      self.titleDict = {}
      pat = re.compile('(\([0-9\-]*\))')
      for title in self.titles:
          for m in pat.finditer(title[0]):
            newTitle = title[0][:m.start()-1]
            date = m.group()
            date = date[1:-1] #comment this out if you want parentheses
            if newTitle in self.titleDict:
                self.titleDict[newTitle].append(date)
            else:
                self.titleDict[newTitle] = [date]
            
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      tempSentiment = dict(reader)
      self.sentiment = {}
      for key, value in tempSentiment.iteritems():
          self.sentiment[self.p.stem(key)] = value
    


    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""
      self.ratings[(self.ratings < self.threshold) & (self.ratings != 0.0)] = -1.0
      self.ratings[self.ratings >= self.threshold] = 1.0
      self.ratings = np.array(self.ratings)

    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      dis = 0.0
      uTotal = 0.0
      vTotal = 0.0
      combTotal = 0.0
      for i, val1 in enumerate(u):
          uTotal += u[i] * u[i]
          vTotal += v[i] * v[i]
          combTotal += u[i] * v[i]
      return combTotal/((uTotal*vTotal)**.5)
  
    def recommend(self, u):
        """Generates a list of movies based on the input vector u using
        collaborative filtering"""
        watched = np.where(u != 0.0)[0]
        
        watched_movies = self.ratings[watched]
        norm = np.matmul(np.linalg.norm(self.ratings, axis=1).reshape(-1, 1),
                         np.linalg.norm(watched_movies, axis=1).reshape(1, -1))
        numer = np.matmul(self.ratings, watched_movies.T)
        similarities = [[numer[i, j] / norm[i, j] if norm[i, j] != 0 else 0.0 for j in range(len(watched))] for i in range(len(u))]
        rankings = np.argsort(np.sum(similarities, axis=1))
        rankings = [self.titles[ranking][0] for ranking in rankings if ranking not in watched]
        return rankings


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
