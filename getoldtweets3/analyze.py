#!/usr/bin/python3

import csv
import time
import sys
from datetime import datetime
from datetime import timedelta
import dateutil.parser as dp
import string
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def main():
    readDir        = sys.argv[1]
    search         = sys.argv[2]
    retweetWeight  = float(sys.argv[3])
    favoriteWeight = float(sys.argv[4])

    scores = {
        'Date': [],
        'TextBlob': [],
        'MasterDict': [],
        'DateInt':  [], # just map date to int for linear regression
        'TextBlobFit':  [],
        'MasterDictFit': []
    }
    dateInt = 0
    for filename in sorted(os.listdir(readDir)):
        filepath=os.path.join(readDir, filename)
        date=filename.replace('tweets-','').replace('-2020.csv','')
        tweets = []
        headers = []
        with open(filepath) as csvfile:
            linereader = csv.reader(csvfile, delimiter=',')
            for header in next(linereader):
                headers.append(header)
            for row in linereader:
                rowDict = {}
                for i in range(len(headers)):
                    rowDict[headers[i]] = row[i]
                tweets.append(rowDict)
        
        scores['Date'].append(date)
        scores['DateInt'].append(dateInt)
        dateInt += 1
        for sentiment in ['TextBlob', 'MasterDict']:
            tweets.sort(key = lambda tweet: tweet[sentiment])
            weightKey  = sentiment+'Weight'
            weightSum  = 0.0
            denominator = 0.0
            for i in range(len(tweets)):
                weight = 1.0 + float(tweets[i]['Retweets'])  * retweetWeight \
                             + float(tweets[i]['Favorites']) * favoriteWeight
                weightSum   += float(tweets[i][sentiment])   * weight
                denominator += weight
                tweets[i][weightKey] = weightSum
            scores[sentiment].append(weightSum / denominator)

    # get fit lines
    for sentiment in ['TextBlob', 'MasterDict']:
        model = LinearRegression()
        xs = np.array(scores['DateInt']).reshape((-1,1))
        model.fit(xs, scores[sentiment])
        fitKey = sentiment + 'Fit'
        scores[fitKey] = model.predict(xs).reshape(-1)

    # plot
    fig, ax = plt.subplots()
    fig.suptitle("Sentiment for Tweets Containing '{}' vs Time".format(search))
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score")
    ax.plot_date(scores['Date'], scores['TextBlob'], 'ro', label='TextBlob Score')
    ax.plot_date(scores['Date'], scores['TextBlobFit'], 'r',\
        linestyle='solid', linewidth=0.5)
    ax.plot_date(scores['Date'], scores['MasterDict'], 'bo', label='MasterDict Score')
    ax.plot_date(scores['Date'], scores['MasterDictFit'], 'b',\
        linestyle='solid', linewidth=0.5)
    ax.axvline('05-11', ymin=0.2, ymax=1, label='3rd BTC Halving', color='black')
    ax.legend(loc=4, fontsize=10)
    plt.xticks(rotation=45)
    spacing = 5
    idx = 0
    for label in ax.xaxis.get_ticklabels():
        if (idx % spacing):
            label.set_visible(False)
        idx += 1

    plt.show()


if __name__ == "__main__":
    main()
