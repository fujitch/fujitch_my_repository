# -*- coding: utf-8 -*-

'''
指定した人数以上の日本人のツイッターアカウントを取得し、それぞれのツイートを２００件ずつ取得。
ユーザー１人１人のリストをtweetsとしてtweetsのリストをpickleとして出力
'''

from requests_oauthlib import OAuth1Session
import json
import datetime, time, sys
import twitkey
from random import randint
import unicodedata
import pickle

CONSUMER_KEY = twitkey.twkey['cons_key']
CONSUMER_SECRET = twitkey.twkey['cons_sec']
ACCESS_TOKEN_KEY = twitkey.twkey['accto_key']
ACCESS_TOKEN_SECRET = twitkey.twkey['accto_sec']

session = OAuth1Session(CONSUMER_KEY,
                        CONSUMER_SECRET,
                        ACCESS_TOKEN_KEY,
                        ACCESS_TOKEN_SECRET)

def checkLimit(session):
        '''
        回数制限を取得
        '''
        unavailableCnt = 0
        while True:
            url = "https://api.twitter.com/1.1/application/rate_limit_status.json"
            res = session.get(url)
            if res.status_code == 503:
                # 503 : Service Unavailable
                if unavailableCnt > 10:
                    raise Exception('Twitter API error %d' % res.status_code)
 
                unavailableCnt += 1
                print ('Service Unavailable 503')
                waitUntilReset(time.mktime(datetime.datetime.now().timetuple()) + 30)
                continue
 
            unavailableCnt = 0
 
            if res.status_code != 200:
                raise Exception('Twitter API error %d' % res.status_code)
            return json.loads(res.text)
 
def waitUntilReset(reset):
        '''
        reset 時刻まで sleep
        '''
        seconds = reset - time.mktime(datetime.datetime.now().timetuple())
        seconds = max(seconds, 0)
        print ('\n     =====================')
        print ('     == waiting %d sec ==' % seconds)
        print ('     =====================')
        sys.stdout.flush()
        time.sleep(seconds + 10)  # 念のため + 10 秒
    
def getUsers(session):
    '''
    100件ランダムにユーザリストとして取得
    '''
    idList = ''
    for i in range(100):
        user_id = randint(1, 500000000)
        if idList == '':
            idList = str(user_id)
        else:
            idList = idList + ',' + str(user_id)
    url = 'https://api.twitter.com/1.1/users/lookup.json'
    params = {'user_id':idList}
    try:
        res = session.get(url, params = params)
        if res.status_code != 200:
            return None
        return json.loads(res.text)
    except:
        return None
    
def getTweetsByUserId(user_id, session, count):
    '''
    ユーザーのツイートをuser_idから取得
    '''
    url = 'https://api.twitter.com/1.1/statuses/user_timeline.json'
    params = {'user_id':user_id, 'count':count}
    try:
        res = session.get(url, params = params)
        if res.status_code != 200:
            return None
        return json.loads(res.text)
    except:
        return None
   
def is_japanese(string):
    '''
    テキストにひらがな、カタカナを含んでいるかどうか
    '''
    for ch in string:
        try:
            name = unicodedata.name(ch) 
        except:
            return False
        if "HIRAGANA" in name \
        or "KATAKANA" in name:
            return True
    return False
    

userList = []
count = 0
# まず日本人ユーザーを取得
# 取得したい人数設定
getNum = 1000
# 日本人ユーザー取得処理
while count < getNum:
    
    # api制限チェック
    limit = checkLimit(session)
    limit = limit['resources']
    limit = limit['users']
    limit = limit['/users/lookup']
    # reset時刻までにremaining回apiを叩けるという意味
    remaining = limit['remaining']
    reset = limit['reset']
    print('ユーザー取得')
    print('API制限:')
    print(remaining)
    # remainingが10以下の時はsleep
    if remaining - 10 < 1:
        waitUntilReset(reset)
        continue
    for i in range(remaining - 5):
        # userリスト取得
        users = getUsers(session)
        if users == None:
            continue
        for user in users:
            if is_japanese(user['description']) == False:
                if not user['time_zone'] in [u'Osaka', u'Tokyo', u'Sapporo']:
                    continue
            if user['protected'] == True:
                continue
            userList.append(user)
            count += 1
        print("%f人取得" % (count))
    # 制限解除までスリープ
    waitUntilReset(reset)


getCount = 0
tweetsList = []
# ツイート取得
while True:
    # api制限チェック
    limit = checkLimit(session)
    limit = limit['resources']
    limit = limit['statuses']
    limit = limit['/statuses/user_timeline']
    # reset時刻までにremaining回apiを叩けるという意味
    remaining = limit['remaining']
    reset = limit['reset']
    print('ツイート取得')
    print('API制限:')
    print(remaining)
    # remainingが10以下の時はsleep
    if remaining - 10 < 1:
        waitUntilReset(reset)
        continue
    if len(userList) <= getCount + remaining - 5:
        for i in range(getCount,len(userList)):
            user = userList[i]
            tweets = getTweetsByUserId(user['id'], session, 200)
            if tweets != None:
                tweetsList.append(tweets)
        break
    if len(userList) > getCount + remaining - 5:
        for i in range(remaining-5):
            user = userList[getCount+i]
            tweets = getTweetsByUserId(user['id'], session, 200)
            if tweets != None:
                tweetsList.append(tweets)
    getCount += remaining - 5
    print("%fセット取得" % (getCount))
    # 制限解除までスリープ
    waitUntilReset(reset)
    
with open('random_0609.pickle', mode='wb') as f:
    pickle.dump(tweetsList, f, protocol=2)