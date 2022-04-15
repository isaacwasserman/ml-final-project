For my final project, I will explore the use of generative adversarial networks (GANs) for data augmentation. Although GANs are most popularly used for domain transfer, the same basic architecture is also a good candidate for a data augmentation strategy called hallucination in which new (fake) training examples based on a small number of existing (real) examples. Though theoretically, the addition of these fake examples should not improve the performance of a discriminative model (since they can be no more representative of the true distribution than the examples they are based on), empirical studies have observed accuracy improvements on multi-class classification of up to 13% on benchmark low-resource datasets such as Omniglot (CITATION NEEDED). Data hallucination has been shown to improve performance, even without GANs; for example, randomized stem hallucination can improve the accuracy of morphological inflection models on low-resource languages by up to 76% (CITATION NEEDED). Surely, there is something to this strategy.

I will start by implementing a simple GAN for the hallucination of a simple image dataset like MNIST; this process will allow me to become familiar with the technique in a medium that I can easily interpret. I then plan to test GAN based hallucination on the morphological inflection model built by Anastasopoulos and Neubig, replacing their largely randomized method with one that better models the phonotactics of the language. The dataset they used was published as part of the SIGMORPHON 2018 shared task (CITATION NEEDED), and it includes subsets for 58 language pairs\footnote{The Anastasopoulos and Neubig model was built to be trained first on a high-resource language and then fine-tuned for a low-resource language with similar genealogy or morphological features} and 91 individual languages.

Acknowledging the possibility that this may be more than I can accomplish in the given timeframe, I may end up needing to forgo the application of GAN hallucination to the inflection model and instead apply it to a simpler image classification task. In this case, I would first test traditional data augmentation methods and compare the effects to those of GAN hallucination.

Here is my proposed timeline:

   **Between now and April 15th:**

    Learn more about GANs

    Learn more about data augmentation

  **April 16th through May 1st:**

    Implement a simple GAN for fake MNIST example generation

    Collect baseline and random hallucination accuracies for Anastasopoulos and Neubig model

  **May 2nd through May 13th:**

    Implement GAN hallucinator to replace random hallucinator

    Write report
