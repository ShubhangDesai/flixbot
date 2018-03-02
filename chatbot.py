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

      #Kaylie needs all these crazy articles for foreign language article handling:/
      self.articles = ["an", "the", "a", "le", "la", "les", "des", "das", "i", "de", "die", "der", "un", "en", "une", "el", "den", "il", "lo", "los", "las"]

      ##Kaylie needs this as placeholder for partial titles for disambiguate, chosen to not conflict or interfere
      self.PLACEHOLDER_TITLE = "UNKKKPARTIAL"

      ##Koby's variables for cleanly keeping track of state

      ##the state of the bot 'MOVIE', 'REC', 'CLARIFY' (for sentiment), 'REASK' (for movie)
      self.genState = 'MOVIE'
      ##the last emotion mentioned 
      self.emoState = 'NEUTRAL'
      ##the last sentiment
      self.sentState = 0.0
      ##the last movie
      self.movState = 'NONE'
      

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

      self.posSet3 = ['I agree, \"%s\" was a modern masterpiece! ', 'Same, \"%s\" was an instant classic! ', 'Yessss, \"%s\" was life changing! ', 'You have THE BEST opinions on movies, \"%s\" was great! ', 'I feel you,  \"%s\" was just incredible. ', 'Oh yea, \"%s\" was the best movie ever made. ', 'OMG, \"%s\" was soooo sooooo goood. ']
      
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

    def say_something_creepy(self):
        x = randint(0, 5)
        if x != 3:
            return False
        return True

    def is_opposite(self, input):
        if input.split()[0].lower() == 'but' or input.split()[0].lower() == 'however' or input.split()[0].lower() == 'although' or input.split()[0].lower() == 'though':
            return True
        return False

    def is_same(self, input):
        if "also" in input.lower() or "another" in input.lower() or "same" in input.lower() or "similar" in input.lower():
            return True
        return False
        
    def is_continuation(self, input):
        if "it" in input.lower() or "this" in input.lower() or "that" in input.lower() or "film" in input.lower() or "movie" in input.lower():
            return True
        return False

    def is_a_movie(self, title):
        capitalList = self.titleDict.keys()
        lowerList = [t.lower() for t in self.titleDict.keys()]
        date = None
        if ' ' in title:
            lastWord = title.rsplit(' ', 1)[-1]
            match = re.findall('\(([^A-Za-z]*)\)', lastWord)
            if match: ##assume that it found date
                date = match[0]
                title = title.rsplit(' (', 1)[0]
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
        input = input[:first_quote-1] + input[second_quote+1:]
        return movie, input

    def rearrageArt(self, movie, date):
        if len(movie.split()) == 0:
            return movie
        if date:
            firstWord = movie.split()[0]
            lastWordInd = movie.index(movie.split()[-1])
            if firstWord.lower() in self.articles:
                movie = movie[:lastWordInd-1] + ', ' + firstWord + " " + movie[lastWordInd:]
                movie = movie.split(' ', 1)[1]  #after article handling, if needed
        else:
            firstWord = movie.split()[0]
            if firstWord.lower() in self.articles:
                movie = movie + ', ' + firstWord
                movie = movie.split(' ', 1)[1]  #after article handling, if needed
            if movie[0:2].lower() == "l'":
                movie = movie + ', ' +'L\''
                movie = movie[2:]
        return movie

    def starter_extract(self, input):
        movie, input = self.getMovieFromQuotes(input)
        if not movie:
            return None, None, None
        orig_movie = movie
        movie = self.rearrageArt(movie, True)
        return orig_movie, movie, input

    def LD(self, s, t):
        if s == "":
            return len(t)
        if t == "":
            return len(s)
        if s[-1] == t[-1]:
            cost = 0
        else:
            cost = 1

        res = min([self.LD(s[:-1], t) + 1,
                   self.LD(s, t[:-1]) + 1,
                   self.LD(s[:-1], t[:-1]) + cost])
        return res

    def get_closest(self, movie, movie_list, date, capital_list):
        possible_movie, min_dist, min_hist = '', float('Inf'), float('Inf')
        for i, item in enumerate(movie_list):
            if abs(len(item) - len(movie)) <= 3:
                hist = [abs(item.count(let) - movie.count(let)) for let in 'abcdefghijklmnopqrstuvwxyz ']
                hist_diff = sum(hist)
                if hist_diff <= 7:
                    if len(movie) >= 8:
                        if hist_diff <= min_dist and hist_diff <= 5:
                            proper_title = item
                            if self.rearrageArt(possible_movie, False).lower() in movie_list:
                                proper_title = capital_list[movie_list.index(self.rearrageArt(item, False).lower())]
                            proper_title = capital_list[movie_list.index(proper_title.lower())]
                            possible_movie = item
                            min_dist = hist_diff

                            if date in self.titleDict[proper_title]:
                                break
                    else:
                        dist = self.LD(item, movie)
                        if dist < min_dist and dist <= 5:
                            possible_movie = item
                            min_dist = dist

        return possible_movie, min_dist

    def check_partial(self, movie, lowerList):
        r= re.compile(r"^" + re.escape(movie) + r" and the ")
        newList = filter(r.match, lowerList)
        if len(newList)>0: return True
        r= re.compile(r"^" + re.escape(movie) + r" ?[0-9]?:")
        newList = filter(r.match, lowerList)
        if len(newList)>0: return True
        return False

    def return_readable(self, movie):
        index = movie.rfind(',')
        if movie[index+2:].lower() in self.articles: return movie[index+2:] + " "+ movie[0: index]
        if movie[index+2:] == "L\'": return movie[index+2:] + movie[0: index]
        else: return movie

    def check_foreign(self, movie, titleList):
        #English part
        changed=False
        formatted_movie = self.rearrageArt(movie, None)
        if formatted_movie!=movie: changed=True
        r= re.compile(r"^" + re.escape(formatted_movie) + r" ?\(.*\)")
        newList = filter(r.match, titleList)
        if len(newList)==1:
            whole_title, whole_title_read = str(newList[0]), str(newList[0])
            if changed: whole_title_read = movie+ whole_title[len(movie)+1:]
            indx_alias = whole_title_read.find('(')
            if whole_title_read.find('(a.k.a.')>0: Is_aka = len('(a.k.a. ')
            else: Is_aka = 0
            if indx_alias>0:
                alias_fixed = self.return_readable(whole_title_read[indx_alias+Is_aka: whole_title_read.find(')')])
                whole_title_read = whole_title_read[:whole_title_read.find('(')+Is_aka] + alias_fixed + whole_title_read[whole_title_read.find(')'):]
            return whole_title_read, whole_title, True
        #Foreign    
        r= re.compile(r".*\(( ?a.k.a. ?)?"+ re.escape(formatted_movie) + r"\)")
        newList = filter(r.match, titleList)
        if len(newList)==1: 
            whole_title, whole_title_read = str(newList[0]), str(newList[0])
            eng_title = whole_title[:whole_title.find(' (')]
            eng_title = self.return_readable(eng_title)
            if changed and whole_title_read.find('(a.k.a.')>0: whole_title_read = eng_title + " (a.k.a. "+movie+whole_title_read[whole_title_read.find(')'):]
            elif changed: whole_title_read = eng_title + " ("+movie+whole_title_read[whole_title_read.find(')'):]
            else: whole_title_read = eng_title + whole_title[whole_title.find(' ('):]
            return whole_title_read, whole_title, True
        return None, None, False

    def extract_movie(self, input):
        capitalList = self.titleDict.keys()
        lowerList = [t.lower() for t in self.titleDict.keys()]

        if input.count('\"') >= 2:
            movie, input = self.getMovieFromQuotes(input)
            orig_movie = movie
            match = re.findall('\(([^A-Za-z]*)\)', movie)
            date = None
            if match: # assume that it found date
                date = match[0]
                movie = movie.rsplit(' (', 1)[0]
                orig_movie = movie.rsplit(' (', 1)[0]
            if movie.lower() not in lowerList and self.rearrageArt(movie, False).lower() not in lowerList:
                #when you uncomment this for spellcheck can u make sure it returns original title as movie if not-spellchecked please?:) 
                if self.check_partial(movie.lower(), lowerList):
                    return movie, self.PLACEHOLDER_TITLE, None, None
                orig_name, movie_name, found = self.check_foreign(movie, capitalList)
                if found: 
                    return orig_name, movie_name, input, date
                movie, dist = self.get_closest(movie.lower(), lowerList, date, capitalList)
                if movie.lower() not in lowerList: ##TEMPFIXNUM1
                    return None, None, None, None  
                else:
                    movie = capitalList[lowerList.index(movie.lower())]
                    orig_movie = self.return_readable(movie)
            else:
                if self.rearrageArt(movie, False).lower() in lowerList:
                    movie = capitalList[lowerList.index(self.rearrageArt(movie, False).lower())]
                movie = capitalList[lowerList.index(movie.lower())]
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
                        titleTest = titleTest.rstrip(string.punctuation)
                        fullTitle, date = self.is_a_movie(titleTest)
                    if fullTitle:
                        lastWord = oldTitle.rsplit(' ', 1)[-1]
                        match = re.findall('\(([^A-Za-z]*)\)', lastWord)
                        if match: ##assume that it found date
                            date = match[0]
                            oldTitle = oldTitle.rsplit(' (', 1)[0]
                        oldTitle = oldTitle.rstrip('?:!.,;').title()
                        if m.start() + len(titleTest) >= len(input):
                            return oldTitle, fullTitle, input[:m.start()-1], date
                        else:
                            return oldTitle, fullTitle, input[:(0 if m.start()-1 < 0 else m.start()-1)] + input[m.start()+len(titleTest):], date
                    titleTest = oldTitle
        return None, None, input, None
        

    def extract_sentiment(self, input):
        pos_neg_count = 0
        inv = 1
        mult = 1
        for word in input.split(' '):
            word = word.rstrip(string.punctuation)
            word = word.lower()
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
        ##if there's no sentiment see if there's sentiment from last run to contrast/compare with
        if pos_neg_count == 0.0 and self.sentState != 0.0:
            if self.genState == "REASK":
                pos_neg_count = self.sentState
                self.genState = "MOVIE"
            elif self.is_same(input):
                pos_neg_count = self.sentState
            elif self.is_opposite(input):
                pos_neg_count = self.sentState * -1
        return pos_neg_count

    def get_movie_and_sentiment(self, input):
        if self.is_turbo == True:
            orig_movies, movies, sentiments, dates = [], [], [], []
            while input != "":
                orig_movie, movie, input_removed, date = self.extract_movie(input)
                if not movie:
                    break
                #when there is a title that doesn't exist but is a correct partial starting title for a few movies
                if movie == self.PLACEHOLDER_TITLE: 
                    movies.append(movie)
                    orig_movies.append(orig_movie)
                    dates.append(None)
                    sentiments.append(0.0)
                # if (not input or not movie) and (not self.genState == "CLARIFY" or not self.is_continuation(input)):
                #     break

                joins = [' and ', ' but ', ' or ']
                first_join, first_idx = '', float('Inf')
                for join in joins:
                    idx = input_removed.find(join)
                    if idx != -1 and idx < first_idx:
                        first_join, first_idx = join, idx

                clauses = [input_removed] if first_join == '' else input_removed.split(first_join)
                input = '' if first_join == '' else first_join.join(clauses[1:])

                features = clauses[0]
                if filter(str.isalnum, features) == '':
                    movies.append(movie)
                    sentiments.append(sentiments[-1])
                    dates.append(date)
                    orig_movies.append(orig_movie)
                    continue
                elif filter(str.isalnum, features) == 'not':
                    movies.append(movie)
                    sentiments.append(-1 if sentiments[-1] == 1 else 1)
                    dates.append(date)
                    orig_movies.append(orig_movie)
                    continue

                pos_neg_count = self.extract_sentiment(features)
                movies.append(movie)
                sentiments.append(pos_neg_count)
                dates.append(date)
                orig_movies.append(orig_movie)

            return orig_movies, movies, sentiments, dates
        else:
            orig_movie, movie, input = self.starter_extract(input)
            if not movie and not input:
                return None, None, None

            if movie not in self.titleSet:
                return "NO_TITLE", "NO_TITLE", 0.0

            pos_neg_count = self.extract_sentiment(input)

            if pos_neg_count > 0.0:
                pos_neg_count = 1.0
            elif pos_neg_count < 0.0:
                pos_neg_count = -1.0
            else:
                pos_neg_count = 0.0

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
      if self.is_turbo:
          return self.getCreativeResponse(sentiment)
      else:
          bigPosSet = []
          bigNegSet = []
          bigPosSet.extend(self.posSet + self.posSet2 + self.posSet3)
          bigNegSet.extend(self.negSet + self.negSet2 + self.negSet3)
          if sentiment == 0.0:
              response = random.sample(self.neutralSet, 1)[0]
          else:
              if sentiment > 0:
                  response = random.sample(bigPosSet, 1)[0]
              else:
                  response = random.sample(bigNegSet, 1)[0]
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
            self.genState = 'CLARIFY'
            response = random.sample(self.neutralSet, 1)[0]
            if sentiment > 0 and self.emoState == 'happy':
                response += "Your love for this movie explains why you feel so happy. "
            elif sentiment < 0 and (self.emoState == "angry" or self.emoState == "upset"):
                response += "Your hatred for this film explains why you're so " + self.emoState + " and salty. "
                    
        return response

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

    ##IDEASFORSHUBHANG
    def getFluResponse(self, input):
        ##Replace "You are" with "I am" (ex. "Screw You" becomes "Screw me?")
        ##Replace "I" with "you" (ex. "I am stupid" becomes "You are stupid?")
        ##Respond to "How..." with "I don't know how..."
        ##Respond to "What is..." with "...is whatever you want it to be"
        ##etc etc
        pass

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
          orig_movies, movies, textSentiments, dates = self.get_movie_and_sentiment(input)
          sentiments = []
          for textSentiment in textSentiments:
              if textSentiment > 0:
                  sentiments.append(1.0)
              elif textSentiment < 0:
                  sentiments.append(-1.0)
              else:
                  sentiments.append(0.0)
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
              if not emotion_index and len(movies)==0: 
                  response = "\n"
                  if input[-1] in string.punctuation and input[-1]!="\"": response = input[:-1] + "?"
                  else: response = input + "?"
                  response+= " Sorry, I don\'t think I understand. If you mentioned a movie title, could you try repeating it? "
                  self.sentState = self.extract_sentiment(input)
                  self.genState = 'REASK'
              elif emotion_index:
                  self.emoState = emotions[emotion_index[0]]
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
              if len(movies)==0:
                  if self.genState != 'REASK':
                      self.genState = 'MOVIE'
                  response += "Now, could you tell me about a movie that you have seen?"
              elif movies == 'NO_TITLE':
                  response = 'Sorry, I\'m not familiar with that title.'
                  self.sentState = 0.0
              else:
                  for textSentiment, sentiment, orig_movie, movie, date in zip(textSentiments, sentiments, orig_movies, movies, dates):
                      full_movie = None
                      response +="\n"
                      if not movie:
                         if self.genState == 'CLARIFY':
                             self.update_user_vector(self.movState, self.sentState)
                             response += self.getResponse(textSentiment) % self.movState
                             response += 'Tell me about another movie you have seen.'
                      elif movie == self.PLACEHOLDER_TITLE:
                          response += "Sorry, I'm not sure which \"" + orig_movie + "\" you are refering to since there are multiple ones! Could you please rephrase that?"
                          sentiment = 0.0
                      elif not date:
                          if len(self.titleDict[movie])==1:
                              date = self.titleDict[movie][0]
                              full_movie = orig_movie + " (" + self.titleDict[movie][0] +")"
                              response += "You must be referring to \"" + full_movie +"\"!\n"
                              response += self.getResponse(textSentiment) % full_movie
                          else:
                              full_movie = None
                              response += "Sorry, I'm not sure which \"" + orig_movie + "\" you are refering to since there are multiple ones! Could you please rephrase that?"
                              sentiment = 0.0
                      elif date not in self.titleDict[movie]:
                          response += "Sorry I don't think you provided a correct year! \n"
                          if len(self.titleDict[movie])==1: 
                              full_movie = orig_movie + " (" + self.titleDict[movie][0] +")"
                              response += "You must be referring to " + full_movie +"instead!\n"
                              response += self.getResponse(textSentiment) % full_movie
                          else:
                              full_movie = None
                              response += "Could you please double check and rephrase that?"
                              sentiment = 0.0
                      else:
                          full_movie = orig_movie + " (" + date +")"
                          response += self.getResponse(textSentiment) % full_movie
                          response += 'Tell me about another movie you have seen.'
                      if movie and date:
                          self.movState = movie + " (" + date + ")"
                      self.sentState = sentiment
                      if sentiment != 0.0 and movie and date:
                          self.update_user_vector(movie + " ("+date+")", sentiment) # uses article-handled "X, The" version for title recognition
                          self.genState = 'MOVIE'
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
      return '''For the creative portion, we implemented the following features:
      1. Identifying movies without quotation marks or perfect capitalization
      2. Fine-grained sentiment extraction
      3. Spell-checking movie titles
      4. Disambiguating movie titles for series and year ambiguities
      5. Extracting sentiment with multiple-movie input
      6. Identifying and responding to emotions
      7. Understanding references to things said previously
      8. Speaking very fluently
      9. Alternate/foreign titles
      And note that we integrated quite a few of them with each other, as listed below
      2,6,8: works with 1-9
      3 works with missing year

      '''


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
