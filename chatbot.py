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
import string

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

      ##Koby's variables for cleanly keeping track of state

      ##the state of the bot 'MOVIE', 'REC', 'CLAIFY'
      self.genState = 'MOVIE'
      

      ##Koby's sets for negations/emphasis/strong positive/strong negative
      
      self.negateSet= set([ "ain't", 'aversion', "can't", 'cannot', 'contradictory', 'contrary', 'counteract', 'dispute', 'dispute', "didn't", "don't", 'implausibility', 'impossibility', 'improbability', 'inability', 'incapable', 'incomplete', 'insignificant', 'insufficient', 'negate', 'negation', 'neither', 'never', 'no', 'no', 'no', 'no', 'nobody',  'non', 'none', 'nor', 'not', 'nothing', 'opposite', 'rather', 'unsatisfactory' 'untrue', "won't"])

      self.overstSet = set(['above', 'absolute', 'absolutely', 'abundance', 'abundantly', 'actual', 'actually', 'always', 'assuredly', 'astonishingly', 'atrociously', 'audaciously', 'authentic', 'avid', 'blatant', 'blatantly', 'brilliant', 'brilliantly', 'clear', 'clearly', 'especially', 'countless', 'decadent', 'decadence', 'deluge', 'deep', 'decisive', 'definite', 'essential', 'exceed', 'excess', 'excessive', 'extra', 'extensive', 'extraordinary', 'extreme', 'flagrant', 'focal', 'foremost', 'frequent', 'fundamental', 'gross', 'impressive', 'incontestability', 'indispensable', 'indisputable', 'infallible', 'indispensible', 'invariably', 'irrefutable', 'just', 'literally', 'notable', 'numerous', 'often', 'outright', 'overly', 'particularly', 'peerless', 'really', 'purely', 'rather', 'realiability', 'robust', 'readily', 'so', 'undeniable', 'unfailing', 'unquestionable', 'unwavering', 'uppermost', 'very'])

      self.posSet = set()
      for word in ['good', 'great', 'awesome', 'cool', 'love', 'loved', 'bomb', 'bomb.com', 'sick', 'dope', 'gr8', 'dank', 'favorite', 'best', 'ultimate', 'masterpiece', 'amazing', 'slick', 'fave', '10/10', 'wild', 'shook', 'beautiful', 'stunning', 'wonderful', 'woundrous', 'majestic', 'excellent', 'grand', 'outstanding', 'remarkable', 'chief', 'capital', 'superior', 'suberb', 'berb', 'birb', 'principal', 'royal', 'exalted', 'admire']:
          self.posSet.add(self.p.stem(word))

      self.negSet = set()
      for word in ['awful', 'horrible', 'bad', 'worst', 'broken', 'terrible', 'appalling', 'atrocious', 'depressing', 'dire', 'disgusting', 'dreadful', 'nasty', '0/10', 'unpleasant', 'abominable', 'deplorable', 'gross', 'offensive', 'abhorrent', 'loathsome', 'abhor', 'despise', 'detest', 'loathe', 'shun', 'curse', 'nope', 'nopetrain']:
          self.negSet.add(self.p.stem(word))

      ##Koby's sets for things that the chatbot says

      self.neutralSet = ['I wasn\'t quite sure if you liked \"%s\"...could you phrase that differently? ', 'So did you like \"%s\" or not? ', 'What\'s your opinion on \"%s\"? ', 'You seem to have mixed feelings about \"%s\". Do you mind elaborating? ', 'I can\'t tell if you liked \"%s\". Could you elaborate? ']
      
      self.posSet = ['Glad to hear you liked \"%s\"! ', 'Yea, I\'d give \"%s\" at least a 6/10, it was good. ', 'Yes, \"%s\" was an above average movie. ', 'I too enjoyed \"%s\" ', 'Yeah \"%s\" was a fun movie. ', 'Yep \"%s\" was enjoyable! ', 'Yeah \"%s\" was enjoyable. ']

      self.posSet2 = ['Yea, \"%s\" was a great movie! ', 'Yea, I\'d give \"%s\" at least a 8/10, it was great! ', 'Yea! \"%s\" was definitely very solid. ', 'I also really liked \"%s\". ', 'Totally! \"%s\" was the complete package. ', 'Definitely, \"%s\" was a very good movie. ']

      self.posSet3 = ['I agree, \"%s\" was a modern masterpiece! ', 'Same, \"%s\" was an instant classic! ', 'Yessss, \"%s\" was life changing! ', 'You have THE BEST opinions on movies, \"%s\" was great! ', 'I feel you,  \"%s\" was just incredible. ', 'Oh yea, \"%s\" was the best movie ever made. ', 'OMG, \"%s\" was soooo sooooo goood']
      
      self.negSet = ['Sorry you didn\'t like \"%s\". ', 'Yea, I didn\'t like  \"%s\" either. ', 'Yea, \"%s\" was definitely a bit below average. ', 'I agree, \"%s\" was underwhelming. ', '\"%s\" was not a satifying movie. ', '\"%s\" was definitely not the best movie. ', 'Yea, I\'d give \"%s\" a solid 4/10. ']
      
      self.negSet2 = ['Definitely, \"%s\" was just a bad experience. ', 'Yea...\"%s\" was almost the worst movie I ever saw. ', 'I agree, \"%s\" almost made me cry in a bad way. ', 'I feel you,  \"%s\" was just bad. ', 'You should be a movie critic,  \"%s\" was objectively bad. ', 'Yea, I\'d give \"%s\" a solid 2/10. ']
      
      self.negSet3 = ['Definitely, \"%s\" was just a terrible experience. I was dragged along. ', 'Yea...\"%s\" was the worst movie I ever saw. ', 'I agree, \"%s\" made me cry in a bad way for hours. ', 'I feel you,  \"%s\" was just downright AWFUL. ', 'You should be a movie critic,  \"%s\" was objectively AWFUL. ', 'Yea, I\'d give \"%s\" a solid 0/10. ']
      

      ##Kaylie's sets for emotional words
      self.emotions = []
      #0 angry, 1 fear, 2 sad, 3 happy
      angry =["angry","irate", "mad", "annoyed", "cross", "vexed", "irritated", "indignant", "aggravated", "angered", "bitter", "burning", "embittered", "enraged", "exasperated", "fired up", "frustrated", "fuming", "furious", "inflamed", "outraged", "pissed off","raging", "seething", "steaming", "soreheaded", "stormy"]
      self.emotions.append(list(map(lambda x: self.p.stem(x), angry)))
      scared = ["afraid", "horrified", "frightened", "scared", "terrified", "fearful", "petrified", "terror-stricken", "terror-struck"]
      self.emotions.append(list(map(lambda x: self.p.stem(x), scared)))
      sad = ["sad","urgh", "upset","unhappy", "sorrowful", "dejected", "depressed", "downcast", "miserable", "down", "despondent", "despairing", "disconsolate", "desolate", "wretched", "glum", "gloomy", "doleful", "dismal", "melancholy", "mournful", "woebegone", "forlorn", "crestfallen", "heartbroken", "inconsolable"]
      self.emotions.append(list(map(lambda x: self.p.stem(x), sad)))
      happy=["happy","cheerful", "excited", "surprised", "yay", "cheery", "merry", "joyful", "jovial", "jolly", "jocular", "gleeful", "carefree", "untroubled", "delighted", "smiling", "beaming", "grinning", "good", "lighthearted", "pleased", "contented", "content", "satisfied", "gratified", "buoyant", "radiant", "blithe", "joyous", "beatific"]
      self.emotions.append((list(map(lambda x: self.p.stem(x), happy))))

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
        capitalList = self.titleDict.keys()
        lowerList = [t.lower() for t in self.titleDict.keys()]
        date = None
        if ' ' in title:
            lastWord = title.rsplit(' ', 1)[-1]
            match = re.findall('\(([^A-Za-z]*)\)', lastWord)
            if match: ##assume that it found date
                date = match[0]
                movie = movie.rsplit('(', 1)[0]
                movie += '\"'
        if title.lower() in lowerList:
            capitalTitle = capitalList[lowerList.index(title.lower())]
            if len(self.titleDict[capitalTitle]) == 1:
                date = self.titleDict[capitalTitle][0]
            return capitalTitle, date
        firstWord = title.split()[0]
        if firstWord.lower() == "an" or firstWord.lower() == "the" or firstWord.lower() == "a":
            title = title + ', ' + firstWord
            title = title.split(' ', 1)[1]  #after article handling, if needed
        if title.lower() in lowerList:
            capitalTitle = capitalList[lowerList.index(title.lower())]
            if len(self.titleDict[capitalTitle]) == 1:
                date = self.titleDict[capitalTitle][0]
            return capitalTitle, date
        return None, None

    def getMovieFromQuotes(self, input):
        first_quote = input.find('\"') + 1
        second_quote = first_quote + input[first_quote:].find('\"')
        movie = input[first_quote:second_quote]
        input = input[:first_quote-2] + input[second_quote+1:]
        return movie, input

    def rearrageArt(self, movie, date):
        if date:
            firstWord = movie.split()[0]
            lastWordInd = movie.index(movie.split()[-1])
            if firstWord.lower() == "an" or firstWord.lower() == "the" or firstWord.lower() == "a":
                movie = movie[:lastWordInd-1] + ', ' + firstWord + " " + movie[lastWordInd:]
                movie = movie.split(' ', 1)[1]  #after article handling, if needed
        else:
            firstWord = movie.split()[0]
            if firstWord.lower() == "an" or firstWord.lower() == "the" or firstWord.lower() == "a":
                movie = movie + ', ' + firstWord
                movie = movie.split(' ', 1)[1]  #after article handling, if needed
        return movie

    def starter_extract(self, input):
        movie, input = self.getMovieFromQuotes(input)
        if not movie:
            return None, None, None
        orig_movie = movie
        movie = self.rearrageArt(movie, True)
        return orig_movie, movie, input

    ##returns title and input(with title extracted)
    ##     example: I like Titanic a lot -> Titanic (1997), I like a lot
    ##returns first moive, so consider recalling it w just input
    ##for disambiguate, consider returning flag for input since input is not needed
    ##     example
    def extract_movie(self, input):
        ##if quotes
        if input.count('\"') == 2:
            movie, input = self.getMovieFromQuotes(input)
            orig_movie = movie
            match = re.findall('\(([^A-Za-z]*)\)', movie)
            date = None
            if match: ##assume that it found date
                date = match[0]
                movie = movie.rsplit('(', 1)[0]
                movie += '\"'
            return orig_movie, movie, input, date
        else:
            pat = re.compile('([A-Z1-9])')
            for m in pat.finditer(input):
                titleTest = input[m.start():]
                titleTest += ' '
                while " " in titleTest:
                    titleTest = titleTest.rsplit(' ', 1)[0]
                    oldTitle = titleTest
                    firstWord = titleTest.split()[0]
                    fullTitle, date = self.is_a_movie(titleTest)
                    if not fullTitle: ##test for punctuation
                        titleTest = titleTest.translate(None, string.punctuation)
                        fullTitle, date = self.is_a_movie(titleTest)
                    if fullTitle:
                        if m.start() + len(titleTest) >= len(input):
                            return oldTitle, fullTitle, input[:m.start()-1], date
                        else:
                            return oldTitle, fullTitle, input[:(0 if m.start()-1 < 0 else m.start()-1)] + input[m.start()+len(titleTest):], date
                    titleTest = oldTitle
        return None, None, None, None
                
            
        if input.count('\"') == 2:
            first_quote = input.find('\"') + 1
            second_quote = first_quote + input[first_quote:].find('\"')
            movie = input[first_quote:second_quote]
            input = input[:first_quote-2] + input[second_quote+1:]
            if movie in self.titleSet:
                return movie, input
            elif movie in self.titleDict.keys():
                return movie + " (" + self.titleDict[movie][0] + ")", input
        

    def extract_sentiment(self, input):
        pos_neg_count = 0
        inv = 1
        mult = 1
        for word in input.split(' '):
            word = word.translate(None, string.punctuation)
            if word in self.overstSet:
                mult += 1
            word = self.p.stem(word)
            if word in self.negateSet:
                inv *= -1
            if word in self.posSet:
                pos_neg_count += 1 * inv * mult
            if word in self.negSet:
                pos_neg_count -= 1 * inv * mult
            if word in self.sentiment:
                if self.sentiment[word] == 'pos':
                    pos_neg_count += 1 * inv * mult
                else:
                    pos_neg_count -= 1 * inv * mult
        if "!!" in input:
            pos_neg_count *= 3
        elif "!" in input:
            pos_neg_count *= 2

        return pos_neg_count

    def get_movie_and_sentiment(self, input):
        if self.is_turbo == True:
            orig_movie, movie, input, date = self.extract_movie(input)
            print movie, date
            if date:
                movie = movie + " (" + date + ")"
                print movie
        else:
            orig_movie, movie, input = self.starter_extract(input)
        if not movie and not input:
            return None, None, None
        if movie not in self.titleSet:
            return "NO_TITLE", "NO_TITLE", 0.0

        '''firstWord = movie.split()[0]
        lastWordInd = movie.index(movie.split()[-1])
        if firstWord.lower() == "an" or firstWord.lower() == "the" or firstWord.lower() == "a":
            movie = movie[:lastWordInd-1] + ', ' + firstWord + " " + movie[lastWordInd:]
            movie = movie.split(' ', 1)[1]  #after article handling, if needed'''

        pos_neg_count = self.extract_sentiment(input)

        return orig_movie, movie, float(pos_neg_count)

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
      if sentiment == 0.0:
          response = random.sample(self.neutralSet, 1)[0]
      else:
        if sentiment > 0:
            response = random.sample(self.posSet, 1)[0]
        else:
            response = random.sample(self.negSet, 1)[0]
      return response

    def getCreativeResponse(self, sentiment):
        if sentiment > 3.0:
            response = random.sample(self.posSet3, 1)[0]
        elif sentiment < -3.0:
            response = random.sample(self.negSet3, 1)[0]
        elif sentiment > 1.0:
            response = random.sample(self.posSet2, 1)[0]
        elif sentiment < -1.0:
            response = random.sample(self.negSet2, 1)[0]
        elif sentiment > 0.0:
            response = random.sample(self.posSet, 1)[0]
        elif sentiment < 0.0:
            response = random.sample(self.negSet, 1)[0]
        else:
            response = random.sample(self.neutralSet, 1)[0]

    def get_emotion(self, input):
      #[angry, fear, sad, happy] 0 1 2 3 
        emotion_scores = [0, 0, 0, 0]
        inv = 1
        mult = 1
        for word in input.split(' '):
            word = word.translate(None, string.punctuation)
            if word in self.overstSet:
                mult += 1
            word = self.p.stem(word)
            if word in self.negateSet:
                inv *= -1
            for i, emotion in enumerate(self.emotions):
                if word in emotion:
                    emotion_scores[i] += 1 * inv * mult
        if "!!" in input:
            emotion_scores = [i * 3 for i in emotion_scores]
        elif "!" in input:
            emotion_scores = [i * 2 for i in emotion_scores]
        m = max(emotion_scores)
        if m == 0: return None
        else: return [i for i, j in enumerate(emotion_scores) if j == m]


    def getMovieResponse(self, movie, sentiment, orig_movie):
        if not movie:
            response = 'Sorry, I don\'t understand. Tell me about a movie that you have seen.'
        elif movie == 'NO_TITLE':
            response = 'Sorry, I\'m not familiar with that title.'
        else:
            response = self.getResponse(sentiment) % orig_movie
            if sentiment != 0.0:
                self.update_user_vector(movie, sentiment) # uses article-handled "X, The" version for title recognition
                if self.data_points < 5:
                    response += 'Tell me about another movie you have seen.'
        return response

    def getMovieRecResponse(self):
        response = '\nThat\'s enough for me to make a recommendation.'
        currRec = self.recommend(self.user_vector)
        for rec in currRec:
            if rec not in self.prevRec:
                response += '\nYou should watch ' + rec + '.'
                self.prevRec.add(rec)
                self.numRecs += 1
                break
        response += '\nWould you like to hear another recommendation? (Or enter :quit if you\'re done.)'
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
          orig_movie, movie, textSentiment = self.get_movie_and_sentiment(input)
          if textSentiment > 0:
              sentiment = 1.0
          elif textSentiment < 0:
              sentiment = -1.0
          else:
              sentiment = 0.0
          maybe = False

          ##Maybe and No for recommend
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

          ##Getting movies
          if self.data_points < 5:
              emotion_index = self.get_emotion(input)
              emotions = ["angry", "scared", "upset", "happy"]
              response = ""
              if not emotion_index and not movie: 
                  if input[-1] in string.punctuation: response = input[:-1] + "?" 
                  else: response = input + "?" 
                  response+= " Sorry, I don\'t think I understand. "
              elif emotion_index:
                  if 3 in emotion_index and len(emotion_index)>1: response = "Hmmm seems you're conflicted! "
                  elif 3 in emotion_index and len(emotion_index)==1: response="Glad to hear you're happy! "
                  else: 
                      response = "Sorry to hear you're "
                      first = True
                      for i in emotion_index:
                          if not first: response += " and "
                          response += emotions[i]
                          first = False
                      response += "! Please let me know if I can do anything to help! "
              if not movie:
                  response += "Now, could you tell me about a movie that you have seen?"
              elif movie == 'NO_TITLE':
                  response = 'Sorry, I\'m not familiar with that title.'
              else:
                  response += self.getCreativeResponse(textSentiment) % orig_movie
                  if sentiment != 0.0:
                      self.update_user_vector(movie, sentiment) # uses article-handled "X, The" version for title recognition
                      if self.data_points < 5:
                          response += 'Tell me about another movie you have seen.'
                          
          ##Return recommendation
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
      else:
          orig_movie, movie, sentiment = self.get_movie_and_sentiment(input)
          if self.genState == 'MOVIE':
              response = self.getMovieResponse(movie, sentiment, orig_movie)
              if self.data_points >= 5:
                  self.genState = 'REC'
                  response += self.getMovieRecResponse()
                  return response
          if self.genState == 'REC':
              if input.lower() == 'no' or self.numRecs >= 5:
                  #keeping previous sentiment
                  self.data_points = 0
                  self.numRecs = 0
                  self.genState = 'MOVIE'
                  response = "Okay! Tell me more about movies!"
                  return response
              elif input.lower() != 'yes':
                  response = "I'm not sure what you said! Please answer \'yes\' or \'no\'"
                  return response
              else:
                  response = self.getMovieRecResponse()
              

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
        numer = np.dot(u, v)
        denom = np.sqrt(np.sum(u**2)) * np.sqrt(np.sum(v**2))

        return numer / denom if denom != 0 else 0.0
  
    def recommend(self, u):
        """Generates a list of movies based on the input vector u using
        collaborative filtering"""
        u = np.array(u)
        watched = np.where(u != 0.0)[0]
        scores = np.zeros(self.ratings.shape[0])

        for i in range(self.ratings.shape[0]):
            if i in watched:
                continue

            for j in watched:
                scores[i] += u[j] * self.distance(self.ratings[i], self.ratings[j])

        recommend_rankings = list(np.argsort(scores)[::-1])
        for j in watched:
            recommend_rankings.remove(j)

        recommendations = np.array(self.titles)[np.array(recommend_rankings)]
        recommendations = [recommendation[0] for recommendation in recommendations]

        return recommendations


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
