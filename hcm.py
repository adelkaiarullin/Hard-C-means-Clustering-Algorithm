import matplotlib.pyplot as pl
import numpy as np
import random

class Data():
    def __init__(self, num_classes, num_samples, eps, dataset=None, centers=None):
        """
        Generate dataset to show the method of HCM.
        """
        self.eps = eps
        self.num_classes = num_classes
        self.num_samples = num_samples
        if dataset is None:
            self.dataset = []

            for _ in range(num_classes):
                d1 = random.uniform(4, 8)
                m1 = random.uniform(-50, 50)
                d2 = random.uniform(4, 8)
                m2 = random.uniform(-50, 50)
                self.dataset.extend([[random.gauss(m1, d1), random.gauss(m2, d2)] for _ in range(num_samples)])
                
            self.dataset = np.array(self.dataset)
        else:
            self.dataset = dataset
        if centers is None:
            self.create_centers()
        else:
            self.centers = centers

    def create_centers(self):
        """
        Create random centers of our classes
        """
        self.centers = []

        rand_inxs = [index for index in range(self.num_samples)]
        random.shuffle(rand_inxs)
        for i in rand_inxs[:3]:
            self.centers.append(self.dataset[i])
        
        self.centers = np.array(self.centers)

    def show_data_and_centers(self):
        pl.figure("data")
        size = 50 # size of points at a plot
        pl.scatter(self.dataset[:, 0], self.dataset[:, 1], c=self.get_labels(), s=size)
        
        pl.figure("before")
        colors = np.array([i for i in range(self.num_classes)])
        pl.scatter(self.centers[:, 0], self.centers[:, 1], c=colors, s=size)

        pl.figure("after")
        epoch = self.move_centers()
        pl.scatter(self.centers[:, 0], self.centers[:, 1], c=colors, s=size)
        
        print("Number of iterations is ", epoch)
        pl.show()

    def metrics(self, first, second):
        """
        The method returns euclidean distance between first and second.
        """
        return np.sqrt(np.sum((first - second) ** 2))

    def has_stopped(self, new_centers, old_centers):
        """
        The method determs when we need stop changing coordinates of centers
        """
        for i in range(self.num_classes):
            if self.metrics(new_centers[i], old_centers[i]) > self.eps:
                return False

        return True

    def move_centers(self):
        """
        The methods is moving the centers of classes. 
        """
        epoch  = 0
        while True:
            counter = np.zeros(shape=(self.num_classes, 3))
            epoch += 1

            for sample in self.dataset:
                distance = [self.metrics(sample, center) for center in self.centers]
                min_d = min(distance)
                index = distance.index(min_d)
                counter[index][0] += sample[0]
                counter[index][1] += sample[1]
                counter[index][2] += 1
            
            new_centers = np.array([[e[0] / e[2], e[1] / e[2]]  if e[2] > 0 else [self.centers[i][0], self.centers[i][1]] for i, e in enumerate(counter)])
        
            if self.has_stopped(new_centers, self.centers):
                break
            else:
                self.centers = new_centers
        
        return epoch
    
    def get_labels(self):
        """
        Use the method to get labels of clustered data
        """
        f = lambda x: np.array([self.metrics(x, c) for c in self.centers]).argmin()
        labels = np.array([f(sample) for sample in self.dataset])

        return labels
        


if __name__ == "__main__":
    data_set = np.array([[5, 6, 7, 7, 8, 5, 8, 4, 6, 7],
                        [7, 6, 8, 7, 7, 7, 5, 8, 8, 6]]).T

    data = Data(num_classes = 3, num_samples = len(data_set), eps = 1e-9, dataset=data_set)
    data.show_data_and_centers()
