import util
from FisherLDA import mainFisherLDAtest
from robustFisherLDA import mainRobustFisherLDAtest
import numpy as np
import matplotlib.pyplot as plt
import pickle

def mainSaveData(dataset = 'sonar'):
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    """
    fisher LDA
    """
    fisherAccuracy = np.empty(shape=(0, len(alphas)))
    for i in range(10):
        rightRates = []
        for alpha in alphas:
            rightRate = mainFisherLDAtest(dataset, alpha)
            rightRates.append(rightRate)
        rightRates = np.array(rightRates)
        rightRates = rightRates.reshape(1, len(alphas))
        fisherAccuracy = np.concatenate((fisherAccuracy, rightRates), axis=0)


    """
    robust Fisher LDA
    """
    robustFisherAccuracy = np.empty(shape=(0, len(alphas)))
    for i in range(10):
        rightRates = []
        for alpha in alphas:
            rightRate = mainRobustFisherLDAtest(dataset, alpha)
            rightRates.append(rightRate)
        rightRates = np.array(rightRates)
        rightRates = rightRates.reshape(1, len(alphas))
        robustFisherAccuracy = np.concatenate((robustFisherAccuracy, rightRates), axis=0)


    # compute means and stds
    fisherLDA_means = np.mean(fisherAccuracy, axis=0)
    fisherLDA_stds = np.std(fisherAccuracy, axis=0)
    robustFisherLDA_means = np.mean(robustFisherAccuracy, axis=0)
    robustFisherLDA_stds = np.std(robustFisherAccuracy, axis=0)


    """
    begin plot
    """
    plt.fill_between(alphas, fisherLDA_means - fisherLDA_stds,
                     fisherLDA_means + fisherLDA_stds, alpha=0.1, color="r")
    plt.fill_between(alphas, robustFisherLDA_means - robustFisherLDA_stds,
                     robustFisherLDA_means + robustFisherLDA_stds, alpha=0.1, color="g")
    plt.plot(alphas, fisherLDA_means, 'o-', color="r",
             label="Fisher LDA")
    plt.plot(alphas, robustFisherLDA_means, '*-', color="g",
             label="Robust Fisher LDA")

    plt.legend(loc="best")

    plt.xlabel(r'$\alpha$')
    plt.ylabel('TSA(%)')
    plt.title(dataset)
    plt.grid(True)
    plt.show()
    """
    end plot
    """

    # save the right rates into a json file
    with open('result/' + dataset + 'FisherMean.txt', 'w') as outfile:
        pickle.dump(fisherLDA_means.tolist(), outfile)
    with open('result/' + dataset + 'FisherStd.txt', 'w') as outfile:
        pickle.dump(fisherLDA_stds.tolist(), outfile)
    with open('result/' + dataset + 'RobustFisherMean.txt', 'w') as outfile:
        pickle.dump(robustFisherLDA_means.tolist(), outfile)
    with open('result/' + dataset + 'RobustFisherStd.txt', 'w') as outfile:
        pickle.dump(robustFisherLDA_stds.tolist(), outfile)

def mainReadData(dataset = 'sonar'):

    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # load datas
    with open('result/' + dataset + 'FisherMean.txt', 'r') as infile:
        fisherLDA_means = pickle.load(infile)
        fisehrLDA_means = np.array(fisherLDA_means)
    with open('result/' + dataset + 'FisherStd.txt', 'r') as infile:
        fisherLDA_stds = pickle.load(infile)
        fisherLDA_stds = np.array(fisherLDA_stds)
    with open('result/' + dataset + 'RobustFisherMean.txt', 'r') as infile:
        robustFisherLDA_means = pickle.load(infile)
        robustFisherLDA_means = np.array(robustFisherLDA_means)
    with open('result/' + dataset + 'RobustFisherStd.txt', 'r') as infile:
        robustFihserLDA_stds = pickle.load(infile)
        robustFihserLDA_stds = np.array(robustFihserLDA_stds)


    """
    begin plot
    """
    plt.fill_between(alphas, fisherLDA_means - fisherLDA_stds,
                     fisherLDA_means + fisherLDA_stds, alpha=0.1,
                     color="r")
    plt.fill_between(alphas, robustFisherLDA_means - robustFihserLDA_stds,
                     robustFisherLDA_means + robustFihserLDA_stds, alpha=0.1, color="g")
    plt.plot(alphas, fisherLDA_means, 'o-', color="r",
             label="Fisher LDA")
    plt.plot(alphas, robustFisherLDA_means, '*-', color="g",
             label="Robust Fisher LDA")

    plt.legend(loc="best")

    plt.xlabel(r'$\alpha$')
    plt.ylabel('TSA(%)')
    plt.title(dataset)
    plt.grid(True)
    plt.show()
    """
    end plot
    """

if __name__ == "__main__":
    dataset = ['ionosphere', 'sonar']  # choose the dataset
    dataset = dataset[1]
    #mainSaveData(dataset)
    mainReadData(dataset)






