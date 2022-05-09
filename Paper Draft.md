# Modeling the Graphotactics of Low-Resource Languages Using Sequential GANs

## Task

## GANs

Generative adversarial networks (GANs) are a class of unsupervised machine learning architectures, most commonly used for image generation. These networks consist of a generator and a discriminator that are trained simultaneously* on a set of data representing a class or domain; this domain could be anything from photos of human faces to time series of hourly temperatures. The generator is tasked with producing "fake" examples that are within this domain without ever seeing any real examples from the training set. Meanwhile, the discriminator is fed a combination of fake examples (from the generator) and real examples and is tasked with classifying them as real or fake. The respective goals of the generator and discriminator constitute a zero-sum game, in which the generator is constantly trying to outsmart the discriminator, while the discriminator hones its ability to distinguish between in-domain and out-of-domain examples.

*Technically speaking, the generator and discriminator are most often trained one after another on a repeated basis.

Though GANs are most often applied to image data (as in the popular CycleGAN\cite{cyclegan}, StyleGAN\cite{stylegan}, and DiscoGAN\cite{discogan}), the same logic is also applicable to other types of data. For example, in 2017, Esteban et al. developed a pair of recurrent GANs which they applied to medical time series data \cite{rcgan}, and in 2016, Yu et al. developed a GAN architecture made specifically for sequences and language generation, utilizing techniques from reinforcement learning \cite{seqgan}.

## Anastasopoulos and Neubig, 2019

## Data Augmentation with GANs

## Method

## Results

## Discussion

## Conclusion


