# fearless_tweeter
Illustrating language models by building one to tweet in the style of our president.


The purpose is mainly illustrative. I had fun building it, and I hope it to be straightforward enough to be a good starting point for understanding how these models work. The code builds a simple language model for generating tweets, with minimal preprocessing steps and no leveraging GPUs. 


The size of the internal state is flexbile, but the model is really designed as a toy. Default sizes are included in the code and are small enough to train on my laptop in a hour or two.

The models will work with any csv file that contains a 'text' column, so feel free to experiment with pretty much any dataset.

The training data used is all tweets from @realdonaldtrump over about 9 years, and is uploaded to the repo. Shoutout to bpb27, whose twitter scraping project at https://github.com/bpb27/twitter_scraping was used to get the data for the models.



## Getting Started

Everything is built in python, and the only package requirements are tensorflow for the tensorflow model and pytorch for the pytorch model. 

Each file has an AutoRegressiveModel class, that implements a simple version of the language generating problem. It should be initialized with a path to the csv file storing the training text, and optional model architecture hyperparamters. 

Construct the model by calling the .build_model() method. The model is initizlized slightly differently depending on whether we're training or testing, and if a saved_model_path is passed it will assume we are building for generating new examples.

Train the model by calling .train_model(). This should be given a path to store the weights for the model when training is over. You can also optionally set the batch size and number of epochs for the training procedure, and for the tensorflow model an optional directory for tensorboard can be passed.



