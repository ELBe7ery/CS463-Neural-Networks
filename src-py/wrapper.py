"""
Wrapper around the KNN classifer
"""
import cifar_classify

# Data set directory here, not included at the repo
DATASET_DIR = 'F:\\Handasa\\Computer\\4th 7asbat\\Neural Networks\\labs\\KNN\\cifar-10-batches-py'
NUM_K = 5
MY_CLF = cifar_classify.KNN(NUM_K, dataset_dir=DATASET_DIR)


### RANDOM TEST CASES

# for i in range(5, 70, 7):
#     print MY_CLF.classify(i, debug=True)

MY_CLF.calc_dist_test()