import matplotlib.pyplot as pl
import random

class Data():

    def __init__(self, num_classes, num_samples, eps):
        """
        Generate dataset to show the method of HCM.
        """
        self.eps = eps
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.dataset = []

        for _ in range(num_classes):
            d1 = random.uniform(4, 8)
            m1 = random.uniform(-50, 50)
            d2 = random.uniform(4, 8)
            m2 = random.uniform(-50, 50)
            self.dataset.append([(random.gauss(m1, d1), random.gauss(m2, d2)) for _ in range(num_samples)])
        
        self.create_centers()


    def create_centers(self):
        """
        Create random centers of our classes
        """
        self.centers = []

        for _ in range(self.num_classes):
            i = random.randint(0, self.num_classes - 1)
            j = random.randint(0, self.num_samples - 1)
            self.centers.append(self.dataset[i][j])


    def show_data_and_centers(self):
        pl.figure()
        size = 10 # size of points at plot
        for group in self.dataset:
            x = [e[0] for e in group]
            y = [e[1] for e in group]
            pl.scatter(x, y, s = size)
        
        pl.figure()
        for center in self.centers:
            pl.scatter(center[0], center[1], s = random.randint(15, 25))

        pl.figure()
        epoch = self.move_centers()
        for center in self.centers:
            pl.scatter(center[0], center[1], s = random.randint(15, 25))
        
        print("Number of iterations is ", epoch)
        pl.show()


    def metrics(self, first, second):
        """
        The method returns euclidean distance between first and second.
        """
        return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 ) ** 0.5


    def has_stopped(self, new_centers, old_centers):
        """
        The method determs when we need stop changing coordinates of centers
        """
        for n, o in zip(new_centers, old_centers):
            if self.metrics(n, o) > self.eps:
                return False
        return True
        


    def move_centers(self):
        """
        The methods is moving the centers of classes. 
        """
        epoch  = 0
        while True:
            counter = [[0.0, 0.0, 0] for _ in self.centers]
            epoch += 1

            for group in self.dataset:
                for sample in group:
                    distance = [self.metrics(sample, center) for center in self.centers]
                    min_d = min(distance)
                    index = distance.index(min_d)
                    counter[index][0] += sample[0]
                    counter[index][1] += sample[1]
                    counter[index][2] += 1
            
            new_centers = [(e[0] / e[2], e[1] / e[2]) if e[2] > 0 else 0 for e in counter]
            if self.has_stopped(new_centers, self.centers):
                break
            else:
                self.centers = new_centers
        
        return epoch


if __name__ == "__main__":
    data = Data(num_classes = 10, num_samples = 100, eps = 1e-7)
    data.show_data_and_centers()
