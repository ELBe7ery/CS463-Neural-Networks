"""
Wrapper around the KNN classifer
"""
import cifar_classify

# Data set directory here, not included at the repo
DATASET_DIR = 'F:\\Handasa\\Computer\\4th 7asbat\\Neural Networks\\labs\\KNN\\cifar-10-batches-py'
NUM_K = 1
MY_CLF = cifar_classify.KNN(NUM_K, dataset_dir=DATASET_DIR)


### TEST CASE
print MY_CLF.classify(271)
