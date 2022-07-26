Sorry for the sudden withdrawal of the core code 'Trans.py'. That's because I found that the validation process are not rigorous engough, and I'm upset and don't want to be misleading. 

A revised version will be opened soon.

p.s. Recently I finished an incredibly amazing network, which will be released by Oct. I guess. Good luck.

# EEG-Transformer

## Transformer based Spatial-Temporal Feature Learning for EEG Decoding

At present, people usually use some methods based on convolutional neural networks (CNNs) for Electroencephalograph (EEG) decoding. However, CNNs have limitations in perceiving global dependencies, which is not adequate for common EEG paradigms with a strong overall relationship. Regarding this issue, we propose a novel EEG decoding method that mainly relies on the attention mechanism. The EEG data is firstly preprocessed and spatially filtered. And then, we apply attention transforming on the feature-channel dimension so that the model can enhance more relevant spatial features. The most crucial step is to slice the data in the time dimension for attention transforming, and finally obtain a highly distinguishable representation. At this time, global averaging pooling and a simple fully-connected layer are used to classify different categories of EEG data. Experiments on two public datasets indicate that the strategy of attention transforming effectively utilizes spatial and temporal features. And we have reached the level of the state-of-the-art in multi-classification of EEG, with fewer parameters. As far as we know, it is the first time that a detailed and complete method based on the transformer idea has been proposed in this field. It has good potential to promote the practicality of brain-computer interface (BCI).


![Fig1](https://user-images.githubusercontent.com/19758589/123399456-35aac100-d5d7-11eb-87ea-d480765df6b7.png)


Hope this code can be useful. I would be very appreciate if you cite us in your paper. 😉 
