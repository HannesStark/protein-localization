from utils.preprocess import reduce_embeddings

reduce_embeddings(['data/embeddings/test_t5-encoderOnly.h5'],
                  'data/embeddings/reduced/',
                  ['test_t5_reduced.h5'])