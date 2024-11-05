The butterfly files are as is in hw2

The mammal_closure and noun_closure are generated from https://github.com/facebookresearch/poincare-embeddings, which represents undirected unweighted graphs

The following are normal csv datasets (taken from https://github.com/facebookresearch/PoincareMaps), where the label columns are:
krumsiek11_blobs: labels
Moignard2015: labels
MyeloidProgenitors: cell_type
MyeloidProgenitorsInterv: cell_type
Olsson: labels
Olsson_wo_HSPC2: labels
Paul: labels
Planaria: labels
ToggleSwitch: labels

wordsim_relatednes_goldstandard and wordsim_similarity_goldstandard are undirected weighted graphs taken from http://alfonseca.org/eng/research/wordsim353.html

SimLex-999 is taken from https://fh295.github.io/simlex.html. Probably only the SimLex999 features is relevant

Some social network datasets are included here (unweighted undirected): I currently only downloaded the facebook, GitHub, and twitch ones https://snap.stanford.edu/data/#socnets