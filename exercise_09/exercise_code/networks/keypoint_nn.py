"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

# TODO: Choose from either model and uncomment that line
# class KeypointModel(nn.Module):
class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        self.inputsize = self.hparams["imag_size"] #96x96  1是灰度值  一通道
        self.outputsize = self.hparams["keypoints_size"] #30
        
        k = self.hparams["num_filter_init"]
        k_size_list = [k,k*2,k*2*2,k*2*2*2]
        self.model = nn.Sequential(
            ## Layer 1
            nn.Conv2d(in_channels=1,out_channels=k_size_list[0],kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.1),
            
            ## Layer 2
            nn.Conv2d(in_channels=k_size_list[0],out_channels=k_size_list[1],kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.2),

            ## Layer 3
            nn.Conv2d(in_channels=k_size_list[1],out_channels=k_size_list[2],kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.3),

            ## Layer 4
            nn.Conv2d(in_channels=k_size_list[2],out_channels=k_size_list[3],kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.5),
            
            # Full connect
            nn.Flatten(),
            
            # 1 
            # nn.Linear in_features由输入张量的形状决定，out_features则决定了输出张量的形状 
            nn.Linear(k_size_list[3]*5*5, k_size_list[3]),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # 2  
            nn.Linear(k_size_list[3],self.outputsize),
            nn.Tanh()
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
