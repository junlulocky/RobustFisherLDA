from load import loader as loader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from log import log
from matplotlib import pyplot as plt
import util


def computeMeanVec(X, y, uniqueClass):
    """
    Step 1: Computing the d-dimensional mean vectors for different class
    """
    np.set_printoptions(precision=4)

    mean_vectors = []
    for cl in range(1,len(uniqueClass)+1):
        mean_vectors.append(np.mean(X[y==cl], axis=0))
        log('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
    return mean_vectors

def computeWithinScatterMatrices(X, y, feature_no, uniqueClass, mean_vectors):
    # 2.1 Within-class scatter matrix
    S_W = np.zeros((feature_no, feature_no))
    for cl,mv in zip(range(1,len(uniqueClass)+1), mean_vectors):
        class_sc_mat = np.zeros((feature_no, feature_no))  # scatter matrix for every class
        for row in X[y == cl]:
            row, mv = row.reshape(feature_no,1), mv.reshape(feature_no,1)   # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat                                # sum class scatter matrices
    log('within-class Scatter Matrix: {}\n'.format(S_W))

    return S_W

def computeBetweenClassScatterMatrices(X, y, feature_no, mean_vectors):
    # 2.2 Between-class scatter matrix
    overall_mean = np.mean(X, axis=0)

    S_B = np.zeros((feature_no, feature_no))
    for i,mean_vec in enumerate(mean_vectors):
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(feature_no, 1) # make column vector
        overall_mean = overall_mean.reshape(feature_no, 1) # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    log('between-class Scatter Matrix: {}\n'.format(S_B))

    return S_B

def computeEigenDecom(S_W, S_B, feature_no):
    """
    Step 3: Solving the generalized eigenvalue problem for the matrix S_W^-1 * S_B
    """
    m = 10^-6 # add a very small value to the diagonal of your matrix before inversion
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W+np.eye(S_W.shape[1])*m).dot(S_B))

    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(feature_no, 1)
        log('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
        log('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

    for i in range(len(eig_vals)):
        eigv = eig_vecs[:,i].reshape(feature_no, 1)
        np.testing.assert_array_almost_equal(np.linalg.inv(S_W+np.eye(S_W.shape[1])*m).dot(S_B).dot(eigv),
                                             eig_vals[i] * eigv,
                                             decimal=6, err_msg='', verbose=True)
    log('Eigenvalue Decomposition OK')

    return eig_vals, eig_vecs

def selectFeature(eig_vals, eig_vecs, feature_no):
    """
    Step 4: Selecting linear discriminants for the new feature subspace
    """
    # 4.1. Sorting the eigenvectors by decreasing eigenvalues
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low by the value of eigenvalue
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    log('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        log(i[0])

    log('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i,j in enumerate(eig_pairs):
        log('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

    # 4.2. Choosing k eigenvectors with the largest eigenvalues - here I choose the first two eigenvalues
    W = np.hstack((eig_pairs[0][1].reshape(feature_no, 1), eig_pairs[1][1].reshape(feature_no, 1)))
    log('Matrix W: \n{}'.format(W.real))

    return W

def transformToNewSpace(X, W, sample_no, mean_vectors, uniqueClass):
    """
    Step 5: Transforming the samples onto the new subspace
    """
    X_trans = X.dot(W)
    mean_vecs_trans = []
    for i in range(len(uniqueClass)):
        mean_vecs_trans.append(mean_vectors[i].dot(W))

    #assert X_trans.shape == (sample_no,2), "The matrix is not size of (sample number, 2) dimensional."

    return X_trans, mean_vecs_trans

def computeErrorRate(X_trans, mean_vecs_trans, y):
    """
    Compute the error rate
    """

    """
    Project to the second largest eigenvalue
    """
    uniqueClass = np.unique(y)
    threshold = 0
    for i in range(len(uniqueClass)):
        threshold += mean_vecs_trans[i][1]
    threshold /= len(uniqueClass)
    log("threshold: {}".format(threshold))

    errors = 0
    for (i,cl) in enumerate(uniqueClass):
        label = cl
        tmp = X_trans[y==label, 1]
        # compute the error numbers for class i
        num = len(tmp[tmp<threshold]) if mean_vecs_trans[i][1] > threshold else len(tmp[tmp>=threshold])
        log("error rate in class {} = {}".format(i, num*1.0/len(tmp)))
        errors += num


    errorRate = errors*1.0/X_trans.shape[0]
    log("Error rate for the second largest eigenvalue = {}".format(errorRate))
    log("Accuracy for the second largest eigenvalue = {}".format(1-errorRate))


    """
    Project to the largest eigenvalue - and return
    """
    uniqueClass = np.unique(y)
    threshold = 0
    for i in range(len(uniqueClass)):
        threshold += mean_vecs_trans[i][0]
    threshold /= len(uniqueClass)
    log("threshold: {}".format(threshold))

    errors = 0
    for (i,cl) in enumerate(uniqueClass):
        label = cl
        tmp = X_trans[y==label, 0]
        # compute the error numbers for class i
        num = len(tmp[tmp<threshold]) if mean_vecs_trans[i][0] > threshold else len(tmp[tmp>=threshold])
        log("error rate in class {} = {}".format(i, num*1.0/len(tmp)))
        errors += num


    errorRate = errors*1.0/X_trans.shape[0]
    log("Error rate = {}".format(errorRate))
    log("Accuracy = {}".format(1-errorRate))

    return 1-errorRate, threshold




def plot_step_lda(X_trans, y, label_dict, uniqueClass, dataset, threshold):

    ax = plt.subplot(111)
    for label,marker,color in zip(range(1, len(uniqueClass)+1),('^', 's'),('blue', 'red')):
        plt.scatter(x=X_trans[:,0].real[y == label],
                    y=X_trans[:,1].real[y == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )

    plt.xlabel('LDA1')
    plt.ylabel('LDA2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('Fisher LDA: {} data projection onto the first 2 linear discriminants'.format(dataset))

    # plot the the threshold line
    [bottom, up] = ax.get_ylim()
    #plt.axvline(x=threshold.real, ymin=bottom, ymax=0.3, linewidth=2, color='k', linestyle='--')
    plt.axvline(threshold.real, linewidth=2, color='g')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    #plt.tight_layout
    plt.show()

def mainFisherLDAtest(dataset='sonar', alpha=0.5):
    # load data
    path = dataset + '/' + dataset + '.data'
    load = loader(path)
    [X, y] = load.load()
    [X, y, testX, testY] = util.divide(X, y, alpha)
    X = np.array(X)
    testX = np.array(testX)

    feature_no = X.shape[1] # define the dimension
    sample_no = X.shape[0] # define the sample number

    # preprocessing
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1
    testY = label_encoder.transform(testY) + 1
    uniqueClass = np.unique(y) # define how many class in the outputs
    label_dict = {}   # define the label name
    for i in range(1, len(uniqueClass)+1):
        label_dict[i] = "Class"+str(i)
    log(label_dict)

    # Step 1: Computing the d-dimensional mean vectors for different class
    mean_vectors = computeMeanVec(X, y, uniqueClass)


    # Step 2: Computing the Scatter Matrices
    S_W = computeWithinScatterMatrices(X, y, feature_no, uniqueClass, mean_vectors)
    S_B = computeBetweenClassScatterMatrices(X, y, feature_no, mean_vectors)

    # Step 3: Solving the generalized eigenvalue problem for the matrix S_W^-1 * S_B
    eig_vals, eig_vecs = computeEigenDecom(S_W, S_B, feature_no)

    # Step 4: Selecting linear discriminants for the new feature subspace
    W = selectFeature(eig_vals, eig_vecs, feature_no)

    # Step 5: Transforming the samples onto the new subspace
    X_trans, mean_vecs_trans = transformToNewSpace(testX, W, sample_no, mean_vectors, uniqueClass)


    # Step 6: compute error rate
    accuracy, threshold = computeErrorRate(X_trans, mean_vecs_trans, testY)


    # plot
    #plot_step_lda(X_trans, testY, label_dict, uniqueClass, dataset, threshold)

    return accuracy




if __name__ == "__main__":
    dataset = ['ionosphere', 'sonar']  # choose the dataset
    alpha = 0.6 # choose the train data percentage
    accuracy = mainFisherLDAtest(dataset[1], alpha)
    print accuracy


