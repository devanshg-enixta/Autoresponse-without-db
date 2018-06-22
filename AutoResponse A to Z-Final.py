import pandas as pd
import csv
import unicodedata
import glob
import argparse
import math, re, sys, fnmatch, string
import pandas as pd
import unicodecsv as csv
import enchant
import os
from gensim.utils import to_unicode, smart_open, dict_from_corpus
import pandas as pd
import logging
import argparse
from collections import namedtuple
from gensim.utils import smart_open, to_utf8, tokenize
from nltk.tokenize import word_tokenize
import math
from nltk.util import ngrams
import random,time
import datetime
from itertools import *
parser = argparse.ArgumentParser("Smaartpulse")

parser.add_argument('--input_reviews_file',
                    type=str,
                    help='enter the source review file')
parser.add_argument('--aspect_words_file',
                    type=str)
parser.add_argument('--lexicon_file',
                    type=str)

parser.add_argument('--response_file',
                    type=str)

args = parser.parse_args()
now = datetime.datetime.now()
date = str(now.year) + "-" + str(now.month) + "-" + str(now.day)
path = os.getcwd()
reload(sys)
d = enchant.Dict("en_US")
negative = []
positive = []
neutral = []
compounded = []
total1=[]
rev_id=[]
subject=[]
alt_subject=[]

f = args.lexicon_file

##########################################################################################################################################################################
############################ all the relevant functions ##################################################################################################################
def sentiment(text):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence.
    """
    text = text.lower().replace('"','')
    test =text.split(' ')
    length = len(test)
    #print length
    for i in range (0,len(test)-1):
        if '/' in test[i]:
            test[i]=test[i].split('/')[0]

        if test[i] == "five" or test[i] == "one" or test[i] == "four" or test[i] == "three" or test[i] == "two" or test[i] == "5" or test[i] == "4" or test[i] == "3" or test[i] == "2" or test[i] == "1" or test[i] == "no" or test[i] == "open" :
            if test[i+1] =='star' or test[i+1] == 'stars' or test[i+1] == 'more'or test[i+1] == 'pack'or test[i+1] == 'packet'or test[i+1] == 'package':
                test[i]=test[i] + '-' + test[i+1]
                test [i+1]=''
        if test[i] == "stomach" or test[i] == "loose":
            if test[i+1] =='ace' or test[i+1] == 'motion':
                test[i]=test[i] + '-' + test[i+1]
                test [i+1]=''
        if test[i] == "use":
            if test[i+1] =='less' or test[i+1] == 'full' or test[i+1] == 'les' or test[i+1] == 'lss' or test[i+1] == 'ls':
                test[i]=test[i] + '-' + test[i+1]
                test [i+1]=''

        if test[i] == "die" :
            if test[i+1] =='to':
                test[i]=test[i] + '-' + test[i+1]
                if test[i+2]:
                    test[i]=test[i] + '-' + test[i+2]
                    test [i+2]=''
        try:
            if not d.check(test[i]):
                test[i]=d.suggest(test[i])[0]
        except:
            pass

    text = "  ".join(test)
    text = text.strip()
    wordsAndEmoticons = str(text).split() #doesn't separate words from adjacent punctuation (keeps emoticons & contractions)
#    print wordsAndEmoticons
    text_mod = text #regex_remove_punctuation.sub('', text) # removes punctuation (but loses emoticons & contractions)
 #   print text_mod
    wordsOnly = str(text_mod).split()
 #   print wordsOnly
    # get rid of empty items or single letter "words" like 'a' and 'I' from wordsOnly
    for word in wordsOnly:
        if len(word) <= 1:
            wordsOnly.remove(word)    
    # now remove adjacent & redundant punctuation from [wordsAndEmoticons] while keeping emoticons and contractions
    puncList = [".", "!", "?", ",", ";", ":", "-", "'", "\"", 
                "!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?"] 
    for word in wordsOnly:
        for p in puncList:
            pword = p + word
            x1 = wordsAndEmoticons.count(pword)
            while x1 > 0:
                i = wordsAndEmoticons.index(pword)
                wordsAndEmoticons.remove(pword)
                wordsAndEmoticons.insert(i, word)
                x1 = wordsAndEmoticons.count(pword)
            
            wordp = word + p
            x2 = wordsAndEmoticons.count(wordp)
            while x2 > 0:
                i = wordsAndEmoticons.index(wordp)
                wordsAndEmoticons.remove(wordp)
                wordsAndEmoticons.insert(i, word)
                x2 = wordsAndEmoticons.count(wordp)

    # get rid of residual empty items or single letter "words" like 'a' and 'I' from wordsAndEmoticons
    for word in wordsAndEmoticons:
        if len(word) <= 1:
            wordsAndEmoticons.remove(word)
    word_len= len(wordsAndEmoticons)
    #print wordsAndEmoticons   
    # remove stopwords from [wordsAndEmoticons]
    #stopwords = [str(word).strip() for word in open('stopwords.txt')]
    #for word in wordsAndEmoticons:
    #    if word in stopwords:
    #        wordsAndEmoticons.remove(word)
    
    # check for negation
    negate = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not","no", "nothing", "nowhere", 
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",  
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite","less"]
    def negated(list, nWords=[], includeNT=True):
        nWords.extend(negate)
        #print list
        for word in nWords:
            if word in list:
                #print word
                return True
        if includeNT:
            for word in list:
                if "n't" in word:
                    return True
        if "least" in list:
            i = list.index("least")
            if i > 0 and list[i-1] != "at":
                return True
        return False
        
    def normalize(score, alpha=1):
        # normalize the score to be between -1 and 1 using an alpha that approximates the max expected value 
        normScore = score/math.sqrt( ((score*score) + alpha) )
        return normScore
    
    def wildCardMatch(patternWithWildcard, listOfStringsToMatchAgainst):
        listOfMatches = fnmatch.filter(listOfStringsToMatchAgainst, patternWithWildcard)
        return listOfMatches
        
    
    def isALLCAP_differential(wordList):
        countALLCAPS= 0
        for w in wordList:
            if str(w).isupper(): 
                countALLCAPS += 1
        cap_differential = len(wordList) - countALLCAPS
        if cap_differential > 0 and cap_differential < len(wordList):
            isDiff = True
        else: isDiff = False
        return isDiff
    isCap_diff = isALLCAP_differential(wordsAndEmoticons)
    
    b_incr = 0.293 #(empirically derived mean sentiment intensity rating increase for booster words)
    b_decr = -0.293
    # booster/dampener 'intensifiers' or 'degree adverbs' http://en.wiktionary.org/wiki/Category:English_degree_adverbs
    booster_dict = {"absolutely": b_incr, "amazingly": b_incr, "awfully": b_incr, "completely": b_incr, "considerably": b_incr, 
                    "decidedly": b_incr, "deeply": b_incr, "effing": b_incr, "enormously": b_incr, 
                    "entirely": b_incr, "especially": b_incr, "exceptionally": b_incr, "extremely": b_incr,
                    "fabulously": b_incr, "flipping": b_incr, "flippin": b_incr, 
                    "fricking": b_incr, "frickin": b_incr, "frigging": b_incr, "friggin": b_incr, "fully": b_incr, "fucking": b_incr, 
                    "greatly": b_incr, "hella": b_incr, "highly": b_incr, "hugely": b_incr, "incredibly": b_incr, 
                    "intensely": b_incr, "majorly": b_incr, "more": b_incr, "most": b_incr, "must": b_incr, "particularly": b_incr, 
                    "purely": b_incr,"please": b_incr, "quite": b_incr, "really": b_incr, "remarkably": b_incr, 
                    "so": b_incr,  "substantially": b_incr, "damn": b_incr, 
                    "thoroughly": b_incr, "totally": b_incr, "tremendously": b_incr, 
                    "uber": b_incr, "unbelievably": b_incr, "unusually": b_incr, "utterly": b_incr, 
                    "very": b_incr,"big":b_incr,"help":b_incr,"heck":b_incr,"too":b_incr,"lots":b_incr,"atall":b_incr,
                    
                    "almost": b_decr, "barely": b_decr, "hardly": b_decr, "just enough": b_decr, 
                    "kind of": b_decr, "kinda": b_decr, "kindof": b_decr, "kind-of": b_decr,
                    "less": b_decr, "little": b_decr, "marginally": b_decr, "occasionally": b_decr, "partly": b_decr, 
                    "scarcely": b_decr, "slightly": b_decr, "somewhat": b_decr, 
                    "sort of": b_decr, "sorta": b_decr, "sortof": b_decr, "sort-of": b_decr,"rip": b_decr}
    sentiments = []
 #   print wordsAndEmoticons
    for item in wordsAndEmoticons:
       # print (item)
        v = 0
        i = wordsAndEmoticons.index(item)
        if (i < len(wordsAndEmoticons)-1 and str(item).lower() == "kind" and \
           str(wordsAndEmoticons[i+1]).lower() == "of") or str(item).lower() in booster_dict:
            sentiments.append(v)
            continue
        item_lowercase = str(item).lower()
        
        if  item_lowercase in word_valence_dict:
           # print ("item in valence lexion file--->",item_lowercase)
            v = float(word_valence_dict[item_lowercase])
            #print ("val", v)
            c_incr = 0.733 #(empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
            if str(item).isupper() and isCap_diff:
                if v > 0: v += c_incr
                else: v -= c_incr
            def scalar_inc_dec(word, valence):
                scalar = 0.0
                
                word_lower = str(word).lower()
                if word_lower in booster_dict:
                    #print ("words in scalar-->",word_lower)
                    scalar = booster_dict[word_lower]
                    if valence < 0: scalar *= -1
                    if str(word).isupper() and isCap_diff:
                        if valence > 0: scalar += c_incr
                        else:  scalar -= c_incr
 #               print ("scal-->",scalar) 
                return scalar
            n_scalar = -1.0
            if (i+1)<word_len:
                if str(wordsAndEmoticons[i+1]).lower() in word_valence_dict and negated([wordsAndEmoticons[i]]) :
                    print ("insde")

            #print ("previous-->",wordsAndEmoticons[i-1])                               
            try:
                if i > 0 and str(wordsAndEmoticons[i-1]).lower() not in word_valence_dict and (i+1)<=word_len:
                    s1 = scalar_inc_dec(wordsAndEmoticons[i-1], v)
                    v = v+s1
                    
                    if negated([wordsAndEmoticons[i-1]]):
                        
                        v = v*n_scalar
                        #print ("insie",v)
                
            except:
                if i > 0 and str(wordsAndEmoticons[i-1]).lower() not in word_valence_dict and (i+1)<word_len:
                    s1 = scalar_inc_dec(wordsAndEmoticons[i-1], v)
                    v = v+s1
                    
                    if negated([wordsAndEmoticons[i-1]]):
                        v = v*n_scalar
                    #print ("inside 2",v)
                
            if i > 1 and str(wordsAndEmoticons[i-2]).lower() not in word_valence_dict and (i+1)<=word_len:
                s2 = scalar_inc_dec(wordsAndEmoticons[i-2], v)
                if s2 != 0: s2 = s2*0.95
                v = v+s2
                if wordsAndEmoticons[i-2] == "never" and (wordsAndEmoticons[i-1] == "so" or wordsAndEmoticons[i-1] == "this"): 
                    v = v*1.5                    
                
                elif negated([wordsAndEmoticons[i-2]]):
                    v = v*n_scalar
            if i > 2 and str(wordsAndEmoticons[i-3]).lower() not in word_valence_dict and (i+1)<word_len:
                s3 = scalar_inc_dec(wordsAndEmoticons[i-3], v)
                if s3 != 0: s3 = s3*0.9
                v = v+s3
                if wordsAndEmoticons[i-3] == "never" and \
                   (wordsAndEmoticons[i-2] == "so" or wordsAndEmoticons[i-2] == "this") or \
                   (wordsAndEmoticons[i-1] == "so" or wordsAndEmoticons[i-1] == "this"):
                    v = v*1.25
                elif negated([wordsAndEmoticons[i-3]]): v = v*n_scalar
                
                # check for special case idioms using a sentiment-laden keyword known to SAGE
                special_case_idioms = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "yeah right": -2, 
                                       "cut the mustard": 2, "kiss of death": -1.5, "hand to mouth": -2}
                # future work: consider other sentiment-laden idioms
                #other_idioms = {"back handed": -2, "blow smoke": -2, "blowing smoke": -2, "upper hand": 1, "break a leg": 2, 
                #                "cooking with gas": 2, "in the black": 2, "in the red": -2, "on the ball": 2,"under the weather": -2}
                onezero = "{} {}".format(str(wordsAndEmoticons[i-1]), str(wordsAndEmoticons[i]))
                twoonezero = "{} {}".format(str(wordsAndEmoticons[i-2]), str(wordsAndEmoticons[i-1]), str(wordsAndEmoticons[i]))
                twoone = "{} {}".format(str(wordsAndEmoticons[i-2]), str(wordsAndEmoticons[i-1]))
                threetwoone = "{} {} {}".format(str(wordsAndEmoticons[i-3]), str(wordsAndEmoticons[i-2]), str(wordsAndEmoticons[i-1]))
                threetwo = "{} {}".format(str(wordsAndEmoticons[i-3]), str(wordsAndEmoticons[i-2]))                    
                if onezero in special_case_idioms: v = special_case_idioms[onezero]
                elif twoonezero in special_case_idioms: v = special_case_idioms[twoonezero]
                elif twoone in special_case_idioms: v = special_case_idioms[twoone]
                elif threetwoone in special_case_idioms: v = special_case_idioms[threetwoone]
                elif threetwo in special_case_idioms: v = special_case_idioms[threetwo]
                if len(wordsAndEmoticons)-1 > i:
                    zeroone = "{} {}".format(str(wordsAndEmoticons[i]), str(wordsAndEmoticons[i+1]))
                    if zeroone in special_case_idioms: v = special_case_idioms[zeroone]
                if len(wordsAndEmoticons)-1 > i+1:
                    zeroonetwo = "{} {}".format(str(wordsAndEmoticons[i]), str(wordsAndEmoticons[i+1]), str(wordsAndEmoticons[i+2]))
                    if zeroonetwo in special_case_idioms: v = special_case_idioms[zeroonetwo]
                
                # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
                if threetwo in booster_dict or twoone in booster_dict:
                    v = v+b_decr
            
            # check for negation case using "least"
            if i > 1 and str(wordsAndEmoticons[i-1]).lower() not in word_valence_dict \
                and str(wordsAndEmoticons[i-1]).lower() == "least" and (i+1)<word_len:
                if (str(wordsAndEmoticons[i-2]).lower() != "at" and str(wordsAndEmoticons[i-2]).lower() != "very") :
                    v = v*n_scalar
            elif i > 0 and str(wordsAndEmoticons[i-1]).lower() not in word_valence_dict \
                and str(wordsAndEmoticons[i-1]).lower() == "least" and (i+1)<word_len:
                v = v*n_scalar
       # print (v)
        sentiments.append(v) 
            
    # check for modification in sentiment due to contrastive conjunction 'but'
    if 'but' in wordsAndEmoticons or 'BUT' in wordsAndEmoticons:#
        try: bi = wordsAndEmoticons.index('but')
        except: bi = wordsAndEmoticons.index('BUT')
        for s in sentiments:
            si = sentiments.index(s)
            if si <bi :
                sentiments.pop(si)
                sentiments.insert(si, s*0.8)
            elif si>bi :
                #print ("si-->",si)
                sentiments.pop(si)
                sentiments.insert(si, s*1.2)
    if 'request' in wordsAndEmoticons or 'REQUEST' in wordsAndEmoticons:#
        try: bi = wordsAndEmoticons.index('request')
        except: bi = wordsAndEmoticons.index('request')
        for s in sentiments:
            si = sentiments.index(s)
            if si <bi :
                sentiments.pop(si)
                sentiments.insert(si, s*1.8)
    
               
 #       print ("sent-->",sentiments)
                
    if sentiments:
        
        sum_s = float(sum(sentiments))
        #print sentiments, sum_s
        
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = str(text).count("!")
        if ep_count > 4: ep_count = 4
        ep_amplifier = ep_count*0.292 #(empirically derived mean sentiment intensity rating increase for exclamation points)
        if sum_s > 0:  sum_s += ep_amplifier
        elif  sum_s < 0: sum_s -= ep_amplifier
        
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = str(text).count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3: qm_amplifier = qm_count*0.18
            else: qm_amplifier = 0.96
            if sum_s > 0:  sum_s += qm_amplifier
            elif  sum_s < 0: sum_s -= qm_amplifier
       # print (sum_s)
        compound = normalize(sum_s)
        #print (compound)
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        total = 1
        tot = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) +1) # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) -1) # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        
        if pos_sum > math.fabs(neg_sum): pos_sum += (ep_amplifier+qm_amplifier)
        elif pos_sum < math.fabs(neg_sum): neg_sum -= (ep_amplifier+qm_amplifier)

        neu_count=normalize(neu_count)
        total = (math.fabs(pos_sum) + math.fabs(neg_sum))
        if total == 0 :
            total = 1
#        print pos_sum ," ",neg_sum
        pos = (math.fabs(pos_sum / total))
        neg = (math.fabs(neg_sum / total))
        neu = normalize((neu_count/total))
    else:
        compound = 0.0; pos = 0.0; neg = 0.0; neu = 0.0
#    print neg
    tot = pos+(-1)*neg+neu

        
    s = {"neg" : round(neg, 3), 
         "neu" : round(neu, 3),
         "pos" : round(pos, 3),
         "compound" : round(compound, 4),
         "total" : round(tot,3)}

    return s


def vader(x):
    #print type(str(x))
   # row = list(x)
    try:
        x=x.encode('utf-8')
        sentence = str(x).encode('ascii','ignore').decode('ascii')
        i=0
        temp_p=0
        temp_n=0
        temp_c=0
        lines_list = re.split(r'[.!?]+', str(sentence))
        for sent in lines_list:
            sent=sent.strip().lower()
            if sent == '':
                print ("empty")
                continue
            else:
                #print sent
                ss = sentiment(sent)
                #print (ss)
                if not all(value == 0 for value in ss.values()):
                    i=i+1
                temp_p=temp_p+ss['pos']
                temp_n=temp_n+ss['neg']
                temp_c=temp_c+ss['compound']
        #print ("i -->",i)
       # print temp_c
        #print i
        if i==0:
            i=1
        if len(str(sentence).split(" ")) <4:
            i=i*2
        tc=float(temp_c/float(i))*100
        tp=float(temp_p/float(i))*100
        tn = float(temp_n/float(i))*100
    except:
        tc = 0
    return tc


def get_ngrams(text ):
    list1 =[]
    for i in range(1,len(text.split(' '))):
        n_grams = ngrams(word_tokenize(text), i)
        list1.append([ ' '.join(grams) for grams in n_grams])
    list1=list(chain.from_iterable(list1))
    return list1
def aspect_finder(review_file):
    aspect_out =[]
    for row in review_file.to_dict('records'):
        rcs = 100.0
        dead = ['died','death','dying','hospit','killed','ill','vomit']
        aspect = []
        aspect_sent =[]
        aspect_keyword = []
        flag =0
        #try:
        temp_text = row['review_text'].encode('ascii','ignore')
        text = str(temp_text).lower()
        txt =get_ngrams(text)
        ques = set(txt)&set(question_words) 
        ques = sorted(ques, key = lambda k : txt.index(k))
        if any(ques):
            flag =1
            math.ceil(random.uniform(78.4, 85.9)*100)/100
        die = set(txt)&set(dead) 
        die = sorted(die, key = lambda k : txt.index(k))
        if any(die):
            flag =2
            rcs = math.ceil(random.uniform(95.4, 99.9)*100)/100
        aspects  = aspects_df.aspect_name.unique() 

        for asp in aspects:
            t = aspects_df[aspects_df.aspect_name == asp]
            keyword = t['keyword']
            keyword =[i.replace('\n','').split('\r', 1)[0] for i in keyword]
            list1 = set(keyword)&set(txt) 
            list2 = sorted(list1, key = lambda k : txt.index(k))
            if any(list2) :
                aspect.append(asp)
                aspect_keyword.append(list2)
        aspect_keyword=list(chain.from_iterable(aspect_keyword))
        aspect_keyword=[i.replace('\n','').split('\r', 1)[0] for i in aspect_keyword]
        match = set(aspect_keyword)&set(dont_reply_aspects) 
        match = sorted(match, key = lambda k : aspect_keyword.index(k))
        print match
        if any(match):
            rcs = float(rcs - float(len(match)*.65))
            if len(match) == len(aspect_keyword):
                aspect.append('dont_reply')
                flag = 3
                rcs = 0.0
        aspects_sent  = aspect_sent_df.aspect_name.unique() 

        for asp in aspects_sent:
            t = aspect_sent_df[aspect_sent_df.aspect_name == asp]
            keyword = t['keyword']
            keyword =[i.replace('\n','').split('\r', 1)[0] for i in keyword]
            list1 = set(keyword)&set(txt) 
            list2 = sorted(list1, key = lambda k : txt.index(k))
            if any(list2) :
                aspect.append(asp)
                aspect_sent.append(list2)

        if len(temp_text) < 80 and len(aspect_keyword) < 1:
            #aspects_missed = expected - actual
            aspects_missed = 1 - len(aspect_keyword)
        elif len(temp_text) >= 80 and len(temp_text)< 180 and len(aspect_keyword) < 2:
            aspects_missed = 2 - len(aspect_keyword)
        elif len(temp_text) >= 180 and len(temp_text) < 450 and len(aspect_keyword) < 3:
            aspects_missed = 3 - len(aspect_keyword)
        elif len(temp_text) >= 450 and len(aspect_keyword) < 4:
            aspects_missed = 4 - len(aspect_keyword)
        else:
            aspects_missed = 0
        rcs = rcs - 4*aspects_missed*.5
        if row['reviewer_name'] == 'Pedigree Expert':
            rcs=0
        row['aspect']=aspect
        row['aspect_sentiment']=aspect_sent
        row['aspect_keyword'] = aspect_keyword
        row['flag'] = flag
        score = row['confidence_score']
        if int(score) <= 0:
            row['polarity'] = 'negative'
        if int(score) > 0:
            row['polarity'] = 'positive'
        if int(flag) == 1:
            row['polarity'] = 'neutral'
        row['response_confidence_score'] = rcs
        aspect_out.append(row)
    return aspect_out
##########################################################################################################################################################################
###############################################################################################################################################################################
start = time.time()
#####################################################################################
############################ reading reviews file ###################################
review_file = pd.read_csv(args.input_reviews_file,encoding='utf-8',quoting=csv.QUOTE_ALL)
review_file.dropna(subset=['review_text'],inplace=True) 

#####################################################################################
#####################################################################################

#####################################################################################
####### finding confidence/sentiment score for each review using vader ##############
word_valence_dict = dict(map(lambda (w, m): (w, float(m)), [wmsr.strip().split('\t')[0:2] for wmsr in open(f) ]))
regex_remove_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
review_file['confidence_score'] = review_file.review_text.apply(lambda row: vader(row))
#####################################################################################
#####################################################################################

#####################################################################################
###################### reading aspects and sentiment file ###########################

aspects_file = pd.read_csv(args.aspect_words_file)
sentiment_file = pd.read_csv('meta_words.csv')
replies_dict = pd.read_csv(args.response_file)
dont_reply = aspects_file[aspects_file.lexicon_type == 'ADR']
dont_reply_aspects = dont_reply.keyword.unique()
dont_reply_aspects=[i.replace('\n','').split('\r', 1)[0] for i in dont_reply_aspects]
dont_reply_aspects= filter(None, dont_reply_aspects)
aspects_df = aspects_file[aspects_file.lexicon_type == 'A']
aspect_sent_df  = aspects_file[aspects_file.lexicon_type == 'AS']

#####################################################################################
#####################################################################################


#####################################################################################
###################### finding aspects for each review #############################
aspect_out = aspect_finder(review_file)
aspect_out=pd.DataFrame(aspect_out)

#####################################################################################
#####################################################################################


#####################################################################################
###################### finding response for each review #############################
aspect_out=pd.DataFrame(aspect_out)
aspect_out['response'] = aspect_out.apply(lambda row: replier(row), axis = 1)
aspect_out = aspect_out[aspect_out.response!='']
aspect_out = aspect_out[~aspect_out.reviewer_name.str.contains('Expert')]
aspect_out.to_csv('responses_mars_pedigree_'+str(date)+'_all_headers.txt',sep='~',quoting=csv.QUOTE_ALL,index=False,encoding='utf-8')
final_resp = aspect_out
aspect_out['response_2'] = ""
aspect_out['response_3'] = ""
aspect_out['degree'] = ""
final_resp.id = final_resp.id.astype(int)
cols = ['id','source_review_id','response','response_2','response_3','confidence_score','polarity','degree','response_confidence_score']
final_resp = final_resp[cols]
final_resp.to_csv('responses_mars_pedigree_'+str(date)+'_db_ingestion.txt',sep='~',quoting=csv.QUOTE_ALL,index=False,encoding='utf-8')

#####################################################################################
#####################################################################################
end = time.time()
print "Time taken by code is :-",end-start
