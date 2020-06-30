# pancancer_somatic_autoencoder
The autoencoder architecture used in the paper 'A pancancer somatic mutation embedding using autoencoders' https://link.springer.com/article/10.1186/s12859-019-3298-z.


# Abstract
Background: Next generation sequencing instruments are providing new opportunities for comprehensive
analyses of cancer genomes. The increasing availability of tumor data allows to research the complexity of cancer
disease with machine learning methods. The large available repositories of high dimensional tumor samples
characterised with germline and somatic mutation data requires advance computational modelling for data
interpretation. In this work, we propose to analyze this complex data with neural network learning, a methodology
that made impressive advances in image and natural language processing.
# Results
Here we present a tumor mutation profile analysis pipeline based on an autoencoder model, which is used to
discover better representations of lower dimensionality from large somatic mutation data of 40 different tumor types
and subtypes. Kernel learning with hierarchical cluster analysis are used to assess the quality of the learned somatic
mutation embedding, on which support vector machine models are used to accurately classify tumor subtypes.
# Conclusions
The learned latent space maps the original samples in a much lower dimension while keeping the
biological signals from the original tumor samples. This pipeline and the resulting embedding allows an easier
exploration of the heterogeneity within and across tumor types and to perform an accurate classification of tumor
samples in the pan-cancer somatic mutation landscape.
Keywords: Autoencoder, Kernel learning, Cancer genomics
