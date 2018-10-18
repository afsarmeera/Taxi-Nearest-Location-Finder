{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Importing Libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import math\n",
    "import operator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      latitude  longitude  cluster\n",
      "0    40.719810 -74.002581        0\n",
      "1    40.745164 -73.982519        0\n",
      "2    40.740104 -73.989658        0\n",
      "3    40.751591 -73.974121        0\n",
      "4    40.779422 -73.955341        0\n",
      "5    40.743483 -73.994009        0\n",
      "6    40.742608 -73.992705        0\n",
      "7    40.741191 -73.989663        0\n",
      "8    40.704588 -74.009639        0\n",
      "9    40.826790 -73.949509        0\n",
      "10   40.755275 -73.978806        0\n",
      "11   40.742816 -74.000406        0\n",
      "12   40.760673 -74.003677        0\n",
      "13   40.736676 -73.988910        0\n",
      "14   40.790599 -73.980234        0\n",
      "15   40.752307 -73.971854        0\n",
      "16   40.741862 -73.989434        0\n",
      "17   40.750946 -74.005634        0\n",
      "18   40.747738 -73.985198        0\n",
      "19   40.731698 -73.989227        0\n",
      "20   40.826508 -73.950391        0\n",
      "21   40.742188 -73.987924        0\n",
      "22   40.741408 -73.988438        0\n",
      "23   40.697433 -73.993043        0\n",
      "24   40.778078 -73.954492        0\n",
      "25   40.728172 -74.007522        0\n",
      "26   40.779014 -73.954039        0\n",
      "27   40.776158 -73.823616        1\n",
      "28   40.750623 -73.990042        0\n",
      "29   40.688930 -73.995666        0\n",
      "..         ...        ...      ...\n",
      "339  40.750795 -73.993576        0\n",
      "340  40.750795 -73.993576        0\n",
      "341  40.758505 -73.989143        0\n",
      "342  40.744901 -73.998059        0\n",
      "343  40.748622 -74.038925        2\n",
      "344  40.783222 -73.833446        1\n",
      "345  40.747513 -73.997875        0\n",
      "346  40.718002 -73.990283        0\n",
      "347  40.742851 -74.006374        0\n",
      "348  40.773560 -73.983144        0\n",
      "349  40.749390 -73.982862        0\n",
      "350  40.741198 -74.002392        0\n",
      "351  40.739673 -74.001697        0\n",
      "352  40.744557 -73.990650        0\n",
      "353  40.812188 -73.955376        0\n",
      "354  40.771426 -73.973501        0\n",
      "355  40.765279 -73.816574        1\n",
      "356  40.735219 -74.009949        0\n",
      "357  40.744425 -73.996492        0\n",
      "358  40.731156 -73.988728        0\n",
      "359  40.678093 -73.973312        0\n",
      "360  40.720582 -73.992714        0\n",
      "361  40.753042 -73.989072        0\n",
      "362  40.765262 -73.980033        0\n",
      "363  40.714605 -74.008056        0\n",
      "364  40.707498 -74.002082        0\n",
      "365  40.935400 -74.071369        3\n",
      "366  40.704273 -73.986759        0\n",
      "367  40.762827 -73.973370        0\n",
      "368  40.737535 -73.990302        0\n",
      "\n",
      "[369 rows x 3 columns]\n"
     ]
    }
   ],
   "source": [
    "# Importing data \n",
    "data = pd.read_csv('B:\\y.csv',usecols=['latitude','longitude','cluster'])\n",
    "print(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [],
   "source": [
    " # Defining a function which calculates euclidean distance between two data points\n",
    "def euclideanDistance(data1, data2, length):\n",
    "    distance = 0\n",
    "    for x in range(length):\n",
    "        distance += np.square(data1[x] - data2[x])\n",
    "    return np.sqrt(distance)\n",
    "\n",
    "# Defining our KNN model\n",
    "def knn(trainingSet, testInstance, k):\n",
    " \n",
    "    distances = {}\n",
    "    sort = {}\n",
    " \n",
    "    length = testInstance.shape[1]\n",
    "    \n",
    "    #### Start of STEP 3\n",
    "    # Calculating euclidean distance between each row of training data and test data\n",
    "    for x in range(len(trainingSet)):\n",
    "   \n",
    "        #### Start of STEP 3.1\n",
    "        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)\n",
    "\n",
    "        distances[x] = dist[0]\n",
    "        #### End of STEP 3.1\n",
    " \n",
    "    #### Start of STEP 3.2\n",
    "    # Sorting them on the basis of distance\n",
    "    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))\n",
    "    #### End of STEP 3.2\n",
    " \n",
    "    neighbors = []\n",
    "    \n",
    "    #### Start of STEP 3.3\n",
    "    # Extracting top k neighbors\n",
    "    for x in range(k):\n",
    "        neighbors.append(sorted_d[x][0])\n",
    "    #### End of STEP 3.3\n",
    "    classVotes = {}\n",
    "    \n",
    "    #### Start of STEP 3.4\n",
    "    # Calculating the most freq class in the neighbors\n",
    "    for x in range(len(neighbors)):\n",
    "        response = trainingSet.iloc[neighbors[x]][-1]\n",
    " \n",
    "        if response in classVotes:\n",
    "            classVotes[response] += 1\n",
    "        else:\n",
    "            classVotes[response] = 1\n",
    "    #### End of STEP 3.4\n",
    "\n",
    "    #### Start of STEP 3.5\n",
    "    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)\n",
    "    return(sortedVotes[0][0], neighbors)\n",
    "    #### End of STEP 3.5\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "testSet = [[40.745164, -73.982519]]\n",
    "test = pd.DataFrame(testSet)\n",
    "print(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.0\n"
     ]
    }
   ],
   "source": [
    "#### Start of STEP 2\n",
    "# Setting number of neighbors = 1\n",
    "k = 5\n",
    "#### End of STEP 2\n",
    "# Running KNN model\n",
    "result,neigh = knn(data, test, k)\n",
    "\n",
    "# Predicted class\n",
    "print(result)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1, 48, 196, 311, 60]\n"
     ]
    }
   ],
   "source": [
    "print(neigh)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [],
   "source": [
    "#data1 = pd.read_csv('B:\\y.csv',usecols=['latitude','longitude'])\n",
    "#testSet = data1.loc[data.index[1],]\n",
    "#test = pd.DataFrame(testSet)\n",
    "#test=test.T\n",
    "#test = pd.DataFrame(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [],
   "source": [
    "#test.to_csv(\"q.csv\", index = False, sep=',', encoding='utf-8')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [],
   "source": [
    "#test = pd.read_csv(\"B:\\q.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.0\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
