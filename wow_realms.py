'''
Popular MMORPG World of Warcraft has multiple servers, also known as realms.
Each realm is designated for a type of gameplay (e.g., PvP, PvE) and has two factions
(Horde and Alliance).  Can a realm's gameplay type (specifically PvP) be predicted based on the 
proportion of its Horde players?
'''

from bs4 import BeautifulSoup
import urllib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from math import exp, log

def sigmoid(x):
    return 1 / (1 + exp(-x))

#Open web page and read in data

r = urllib.urlopen('http://www.wowprogress.com/realms/rank/us').read()
soup = BeautifulSoup(r, 'html.parser')

all_lst = []

#Data is already in table form on web page
#There must be a better way than making one massive list of all <td> elements...

trList = soup.findAll('tr')
for tr in trList:
    tdList = tr.findAll('td')
    for td in tdList:
        all_lst.append(td)

table = {'Realm': [], 'PvP': [], 'Horde': [], 'Alliance': []}

for x in range(1079):
    if x % 9 == 1:
        table['Realm'].append(all_lst[x].get_text())
    elif x % 9 == 2:
        table['PvP'].append(all_lst[x].get_text())
    elif x % 9 == 6:
        table['Horde'].append(int(all_lst[x].get_text()))
    elif x % 9 == 7:
        table['Alliance'].append(int(all_lst[x].get_text()))

df = pd.DataFrame(table)

#Group PvE (Player versus Environment) and RP (RolePlay) as '0'
#Group PvP (Player versus Player) and RP-PvP as '1' 

print df['PvP'].unique()
df.replace(['PvP', 'PvE', 'RP-PvP', 'RP'], [1, 0, 1, 0], inplace=True)
df['Horde_pct'] = df['Horde'] / (df['Alliance'] + df['Horde'])

#Split into train and test sets

X = df['Horde_pct']
y = df['PvP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train = np.concatenate((X_train.values.reshape(96, 1), y_train.values.reshape(96, 1)), axis=1)

#Train using logistic regression gradient descent
#Hypotheis: h(x) = 1 / (1 + exp(-(theta0 + theta1 * x)))
#Probability that realm is PvP
#x is proportion of Horde players on realm
#y is 0 for PvE, 1 for PvP
#Yes, the loop is horrible.  Should be using vectorization instead.

theta0 = 0
theta1 = 0
learning_rate = 5
m = float(len(train))

for _ in range(100):
    total0 = 0
    total1 = 0
    total_cost = 0
    for x, y in train:
        total0 += sigmoid(theta0 + theta1 * x) - y
        total1 += (sigmoid(theta0 + theta1 * x) - y) * x
        total_cost += y * log(sigmoid(theta0 + theta1 * x)) + (1 - y) * log(1 - sigmoid(theta0 + theta1 * x))
    theta0 -= learning_rate / m * total0
    theta1 -= learning_rate / m * total1
    cost = -1 / m * total_cost
    print theta0, theta1, cost

#Test

test = np.concatenate((X_test.values.reshape(24, 1), y_test.values.reshape(24, 1)), axis=1)

score = 0
for x, y in test:
    pred = sigmoid(theta0 + theta1 * x)
    if (pred < 0.5 and y == 0) or (pred >= 0.5 and y ==1):
        score += 1

print score / float(len(test))

#Train on full set (should turn logistic regression into a function...)

full_set = df.as_matrix(columns=['Horde_pct', 'PvP'])

theta0 = 0
theta1 = 0
learning_rate = 5
m = float(len(full_set))

for _ in range(100):
    total0 = 0
    total1 = 0
    total_cost = 0
    for x, y in full_set:
        total0 += sigmoid(theta0 + theta1 * x) - y
        total1 += (sigmoid(theta0 + theta1 * x) - y) * x
        total_cost += y * log(sigmoid(theta0 + theta1 * x)) + (1 - y) * log(1 - sigmoid(theta0 + theta1 * x))
    theta0 -= learning_rate / m * total0
    theta1 -= learning_rate / m * total1
    cost = -1 / m * total_cost
    print theta0, theta1, cost

#Add visualization with decision boundary?