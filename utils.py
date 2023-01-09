import cv2
import matplotlib.pyplot as plt
import numpy as np

class Util:
    def save(self, dir, data):
        cv2.imwrite(dir, data)

class Histogram:
    def __init__(self):
        self.flattened_image = np.array([])
        self.shape = ()
        self.totalPixels = 0
        self.intensidades_labels = [str(x) for x in range(256)]

    def read_image(self, dir_img):
        original_img = cv2.imread(dir_img, cv2.IMREAD_GRAYSCALE)
        self.shape = original_img.shape
        self.totalPixels = original_img.size
        self.flattened_image = np.ravel(original_img)

    def calcHistogram(self):
        frequencies = [0]*256
        for p in self.flattened_image:
            frequencies[p] += 1

        return frequencies

    def calcNormalizedHistogram(self):
        normalized_freqs = [0]*256
        freqs = self.calcHistogram()
        for i,v in enumerate(freqs):
            normalized_freqs[i] = float(v)/self.totalPixels

        return normalized_freqs

    def calcAcumulatedHistogram(self):
        acumulated_hist = [0]*256
        normal_hist = self.calcNormalizedHistogram()

        for i, v in enumerate(normal_hist):
            acumulated_hist[i] = v if i == 0 else acumulated_hist[i-1] + v

        return acumulated_hist

    def calcEqualizedHistogram(self):

        freqs = self.calcHistogram()

        intens_max = len(freqs)-1
        equalized_hist = [0]*(intens_max+1)
        img_size = len(self.flattened_image)
        equalized_flat_image = np.array([0]*img_size)
        mapping = []

        for i, v in enumerate(freqs):
            aux = [intens_max*(freqs[x])/self.totalPixels for x in range(i+1)]
            new_value = round(sum(aux))

            equalized_hist[new_value] = freqs[i] if equalized_hist[new_value] == 0 else equalized_hist[new_value] + freqs[i]
            mapping.append(new_value)

            aux_image = np.array([0]*img_size)
            indexes = np.where(self.flattened_image == i)

            for i in indexes[0]: aux_image[i] = new_value

            equalized_flat_image = np.add(equalized_flat_image, aux_image)

        equalized_image = equalized_flat_image.reshape(self.shape)
        mapping = np.array(mapping)
        return equalized_hist, equalized_image, mapping

    def __find_closest(self, arr, val): # Return index
       idx = np.abs(arr - val).argmin()
       return idx

    def __transformationFunction(self, values):
        dir = "./figures/"
        plt.plot(self.intensidades_labels, values)
        plt.xticks(np.arange(0, len(values)+1, 50))
        plt.savefig(dir + "Funcao_transformacao" + ".png", format="png")
        plt.show()

    def calcSpecifiedHistogram(self, z):
        eq_hist, _, eq_mapping = self.calcEqualizedHistogram()

        L = 256
        specified_hist = [0]*L
        spec_mapping = []

        for i, v in enumerate(z):
            aux = [(L-1)*(z[x])/self.totalPixels for x in range(i+1)]
            new_value = round(sum(aux))

            specified_hist[new_value] = z[i] if specified_hist[new_value] == 0 else specified_hist[new_value] + z[i]
            spec_mapping.append(new_value)
        
        self.__transformationFunction(spec_mapping)
        
        new = []
        for i,v in enumerate(eq_mapping):
            closest = self.__find_closest(spec_mapping, v)
            new.append(closest)


        freqs = self.calcHistogram()
        new_hist = [0]*L
        new_flat_image = np.array([0]*len(self.flattened_image))

        for i, v in enumerate(freqs):
            new_hist[new[i]] = freqs[i] if new_hist[new[i]] == 0 else new_hist[new[i]] + freqs[i]

            aux_image = self.flattened_image.copy()
            aux_image[aux_image!=i] = 0
            aux_image[aux_image==i] = new[i]

            new_flat_image = np.add(new_flat_image, aux_image)

        new_image = new_flat_image.reshape(self.shape)
        
        return new_hist, new_image

    def show(self, valores, title, xlabel, ylabel):
        dir = "./figures/"
        plt.bar(self.intensidades_labels, valores)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(np.arange(0, len(valores)+1, 50))
        plt.savefig(dir + title + ".png", format="png")
        plt.show()

class LinearTransformations:
    def __init__(self):
        self.img = np.array([])

    def read_image(self, dir_img):
        self.img = cv2.imread(dir_img, cv2.IMREAD_GRAYSCALE)
    
    def calcLinear(self, c, b):
        return np.array(c*self.img+b, dtype = np.uint8) # float to int

    def calcLogarithm(self, c):
        log_image = c*np.log2(self.img+1)
        return np.array(log_image, dtype = np.uint8) # float to int

    def calcExponential(self, c):
        exp_image = c*np.exp((self.img+1)/(255+1))*(255/np.exp(1))
        return np.array(exp_image, dtype = np.uint8) # float to int