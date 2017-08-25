import math as math
import numpy as np
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import random as random


def euclid_norm(x):
    return math.sqrt(np.dot(x, x.T))

class SOM:
    def __init__(self, x, y, input_len, learn_coef, sigma, exp_cluster):

        self.x = x
        self.y = y
        self.input_len = input_len
        self.learn_coef = learn_coef
        self.sigma = sigma

        self.exp_cluster = exp_cluster
        self.clusters = np.random.rand(exp_cluster, input_len)
        self.labels = []

        for i in range(exp_cluster):
            self.clusters[i] = self.clusters[i] / euclid_norm(self.clusters[i])

        self.weights = np.random.rand(x, y, input_len)

        for i in range(x):
            for j in range(y):
                self.weights[i][j] = self.weights[i][j] / euclid_norm(self.weights[i][j])


    def train(self, data):
        normalized_data = np.zeros(data.shape)

        for i in range(data.shape[0]):
            normalized_data[i] = data[i] / euclid_norm(data[i])

        """ debugging with heatmap gauss neighborhood
        win_location = self.find_winner(normalized_data[0])
        print(win_location)
        neighborhood = self.gauss(win_location, self.sigma)
        print(neighborhood)
        print(neighborhood.argmin())
        print(neighborhood.argmax())
        plt.imshow(neighborhood, cmap='hot', interpolation='nearest')
        plt.show()
        """

        for t in range(data.shape[0]):
            cur_learn_coef = self.decay_function(self.learn_coef, t, data.shape[0])
            cur_sigma = self.decay_function(self.sigma, t, data.shape[0])

            win_location = self.find_winner(normalized_data[t])
            neighborhood = self.gauss(win_location, cur_sigma)

            neighborhood *= cur_learn_coef

            iter = np.nditer(neighborhood, flags=['multi_index'])

            while not iter.finished:
                self.weights[iter.multi_index] += neighborhood[iter.multi_index]*(data[t]-self.weights[iter.multi_index])

                self.weights[iter.multi_index] = self.weights[iter.multi_index] / euclid_norm(self.weights[iter.multi_index])

                iter.iternext()


        #print(self.weights)
        """
        fig, ax = plt.subplots()
        im = ax.imshow(self.weights, cmap=plt.get_cmap('hot'), interpolation='nearest',
                       vmin=0, vmax=1)
        fig.colorbar(im)
        plt.show()
        """

        """
        how about plotting u-matrix?

        umatrix = np.zeros((2*self.x-1, 2*self.y-1))

        for p in range(self.x):
            for q in range(self.y):
                if (p != self.x-1):
                    umatrix[2*p+1][2*q] = self.get_distance(self.weights[p][q], self.weights[p+1][q])
                if (q != self.y-1):
                    umatrix[2*p][2*q+1] = self.get_distance(self.weights[p][q], self.weights[p][q+1])
                if (p != self.x-1) and (q != self.y-1):
                    umatrix[2*p+1][2*q+1] =  self.get_distance(self.weights[p][q], self.weights[p+1][q+1])



        print(umatrix)
        fig, ax = plt.subplots()
        im = ax.imshow(umatrix, cmap=plt.get_cmap('hot'), interpolation='nearest',
                        vmin=0, vmax=1)
        fig.colorbar(im)
        plt.show()
        """
        first = random.randint(0, 49)
        second = random.randint(0, 49)
        third = random.randint(0, 49)
        winning_neuron_a = self.find_winner(load_iris().data[0+first] / euclid_norm(load_iris().data[0+first]))
        win_weight_a = self.weights[winning_neuron_a[0]][winning_neuron_a[1]]
        winning_neuron_b = self.find_winner(load_iris().data[50+second] / euclid_norm(load_iris().data[50+second]))
        win_weight_b = self.weights[winning_neuron_b[0]][winning_neuron_b[1]]
        winning_neuron_c = self.find_winner(load_iris().data[100+third] / euclid_norm(load_iris().data[100+third]))
        win_weight_c = self.weights[winning_neuron_c[0]][winning_neuron_c[1]]

        self.clusters[0] = win_weight_a
        self.labels.append('setosa')
        self.labels.append('versicolor')
        self.labels.append('virginica')
        self.clusters[1] = win_weight_b
        self.clusters[2] = win_weight_c

        shuffled = []
        for i in range(self.x):
            for j in range(self.y):
                shuffled.append(self.weights[i][j])

        random.shuffle(shuffled)
        """
        fig, ax = plt.subplots()
        im = ax.imshow(shuffled, cmap=plt.get_cmap('hot'), interpolation='nearest',
                       vmin=0, vmax=1)
        fig.colorbar(im)
        plt.show()
        """
        counter = 0
        # learning cluster to classify
        for p in range(len(shuffled)):
            # 122, 6666666
            # good parameters
            # 562689
            # 56235789
            cluster_cur_learn_coef = self.decay_function(self.learn_coef, counter, self.x*self.y)/562616
            cluster_cur_sigma = self.decay_function(self.sigma, counter, self.x*self.y)/56165789

            cluster_win_location = self.find_cluster_winner(shuffled[p])
            cluster_neighborhood = self.cluster_gauss(cluster_win_location, cluster_cur_sigma)

            cluster_neighborhood *= cluster_cur_learn_coef

            cluster_iter = np.nditer(cluster_neighborhood, flags=['f_index'])

            while not cluster_iter.finished:
                self.clusters[cluster_iter.index] += cluster_neighborhood[cluster_iter.index] * (
                shuffled[p] - self.clusters[cluster_iter.index])

                self.clusters[cluster_iter.index] = self.clusters[cluster_iter.index] / euclid_norm(
                    self.clusters[cluster_iter.index])

                cluster_iter.iternext()
            counter += 1


        win_cluster_a = self.find_cluster_winner(win_weight_a)

        win_cluster_b = self.find_cluster_winner(win_weight_b)

        win_cluster_c = self.find_cluster_winner(win_weight_c)
        """
        print(win_weight_a)
        print(win_weight_b)
        print(win_weight_c)

        print(win_cluster_a)
        print(win_cluster_b)
        print(win_cluster_c)
        """


    def find_cluster_winner(self, data):
        min_x = None

        min_dist = math.inf
        cur_dist = None
        for i in range(self.exp_cluster):
            cur_dist = self.get_distance(data, self.clusters[i])
            if (min_dist > cur_dist):
                min_dist = cur_dist
                min_x = i

        return min_x

    def cluster_gauss(self, point, sigma):
        down = 2*math.pi*sigma*sigma

        x_t = np.exp(-1*np.power(np.arange(self.exp_cluster)-point, 2)/down)


        return x_t

    def classify(self, data):
        normalize = data / euclid_norm(data)
        first_index = self.find_winner(normalize)
        second_index = self.find_cluster_winner(self.weights[first_index[0]][first_index[1]])
        return self.labels[second_index]

    def decay_function(self, coef, time, max_t):
        return (1+time)/max_t

    @staticmethod
    def get_distance(data, point):
        distance = 0
        for i in range(len(data)):
            distance += (point[i]-data[i])*(point[i]-data[i])
        return math.sqrt(distance)

    def find_winner(self, data):
        min_x = None
        min_y = None

        min_dist = math.inf
        cur_dist = None
        for i in range(self.x):
            for j in range(self.y):
                cur_dist = self.get_distance(data, self.weights[i][j])
                if (min_dist > cur_dist):
                    min_dist = cur_dist
                    min_x = i
                    min_y = j

        return min_x, min_y

    def gauss(self, point, sigma):
        down = 2*math.pi*sigma*sigma

        x_t = np.exp(-1*np.power(np.arange(self.x)-point[0], 2)/down)
        y_t = np.exp(-1*np.power(np.arange(self.y)-point[1], 2)/down)

        return np.outer(x_t, y_t)

class knn:
    def __init__(self, coef):
        self.k = coef

    def train(self, data, classes):
        self.data = data
        self.classes = classes

    def classify(self, sample):
        normalize = sample / euclid_norm(sample)
        distances = np.zeros(self.data.shape[0])

        for t in range(self.data.shape[0]):
            distances[t] = SOM.get_distance(self.data[t], sample)

        index_list = []
        distances_list = []

        set_count = 0
        vers_count = 0
        virg_count = 0

        for q in range(self.k):
            closest = np.argmin(distances)
            index_list.append(closest)
            distances_list.append(distances[closest])
            distances[closest] = math.inf
            if (self.classes[index_list[q]]==0):
                set_count += 1
            elif (self.classes[index_list[q]]==1):
                vers_count += 1
            elif (self.classes[index_list[q]]==2):
                virg_count += 1

        for p in range(self.k):
            distances[index_list[p]] = distances_list[p]

        major_class = np.argmax((set_count, vers_count, virg_count))

        if (major_class==0):
            return 'setosa'
        elif (major_class==1):
            return 'versicolor'
        elif (major_class==2):
            return 'virginica'






if __name__ == "__main__":
    som_accuracy = []
    knn_accuracy = []
    for loop in range(30):
        som = SOM(23, 23, 4, sigma=0.7, learn_coef=0.6, exp_cluster=3)

        iris_dataset = load_iris()
        setosa = iris_dataset.data[0:50]
        versicolor = iris_dataset.data[50:100]
        virginica = iris_dataset.data[100:150]

        random.shuffle(setosa)
        random.shuffle(versicolor)
        random.shuffle(virginica)

        list_train_data = []
        list_classes = []

        for i in range(30):
            list_train_data.append(setosa[i])
            list_train_data.append(versicolor[i])
            list_train_data.append(virginica[i])
            list_classes.append(0)
            list_classes.append(1)
            list_classes.append(2)

        train_data = np.asarray(list_train_data)
        classes = np.asarray(list_classes)
        som.train(train_data)
        knn_classifier = knn(3)
        knn_classifier.train(train_data, classes)

        som_count = 0
        knn_count = 0
        for t in range(20):
            som_set = som.classify(setosa[t+30])
            som_vers = som.classify(versicolor[t+30])
            som_virg = som.classify(virginica[t+30])
            if som_set == 'setosa':
                som_count += 1
            if som_vers == 'versicolor':
                som_count += 1
            if som_virg == 'virginica':
                som_count += 1

            knn_set = knn_classifier.classify(setosa[t + 30])
            knn_vers = knn_classifier.classify(versicolor[t + 30])
            knn_virg = knn_classifier.classify(virginica[t + 30])

            if knn_set == 'setosa':
                knn_count += 1
            if knn_vers == 'versicolor':
                knn_count += 1
            if knn_virg == 'virginica':
                knn_count += 1


        print(som_count/60)
        print(knn_count/60)

        som_accuracy.append(som_count/60)
        knn_accuracy.append(knn_count / 60)
        print(loop)
    print("som accuracy = "+str(np.mean(som_accuracy)))
    print("knn accuracy = "+str(np.mean(knn_accuracy)))
