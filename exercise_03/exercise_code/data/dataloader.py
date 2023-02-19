"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################

        batch_iterator = self.build_batch_iterator(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        batches = []
        for batch in batch_iterator:
            batches.append(batch)
        
        #combine_batch_dicts
        batch_dict = {}
        for batch in batches:
            for data_dict in batch:
                    for key, value in data_dict.items():
                        if key not in batch_dict:
                            batch_dict[key] = []
                            yield batch_dict
                        batch_dict[key].append(value)
       
        # batch_to_numpy
        numpy_batch = {}
        for key, value in batch_dict.items():
            numpy_batch[key] = np.array(value)
        
        yield numpy_batch

        return numpy_batch
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last!                                 #
        ########################################################################

        if(self.drop_last):
            length =  (int)(len(self.dataset)/self.batch_size)
        else:
            length =  (int)(len(self.dataset)/self.batch_size)+1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length

    
    # def build_batches(dataset, batch_size):
    #     batches = []  # list of all mini-batches
    #     batch = []  # current mini-batch
    #     for i in range(len(dataset)):
    #         batch.append(dataset[i])
    #         if len(batch) == batch_size:  # if the current mini-batch is full,
    #             batches.append(batch)  # add it to the list of mini-batches,
    #             batch = []  # and start a new mini-batch
    #     return batches

    
    # def combine_batch_dicts(batch):
    #     batch_dict = {}
    #     for data_dict in batch:
    #         for key, value in data_dict.items():
    #             if key not in batch_dict:
    #                 batch_dict[key] = []
    #             batch_dict[key].append(value)
    #     return batch_dict

    # def batch_to_numpy(batch):
    #     numpy_batch = {}
    #     for key, value in batch.items():
    #         numpy_batch[key] = np.array(value)
    #     return numpy_batch

    def build_batch_iterator(self, dataset, batch_size, shuffle):
        if shuffle:
            index_iterator = iter(np.random.permutation(len(dataset)))  # define indices as iterator
        else:
            index_iterator = iter(range(len(dataset)))  # define indices as iterator

        batch = []
        for index in index_iterator:  # iterate over indices using the iterator
            batch.append(dataset[index])
            if len(batch) == batch_size:
                yield batch  # use yield keyword to define a iterable generator
                batch = []