from utils.preprocess import combine_embeddings

for type in ['cat']:
#for type in ['cat', 'sum', 'avg', 'max']:
    for stage in ['train', 'val', 'test']:
        combine_embeddings('/mnt/project/bio_embeddings/runs/hannes/embed_' + stage + '/embeddings/bert_embeddings/embeddings_file.h5',
                           '/mnt/project/bio_embeddings/runs/hannes/embed_' + stage + '_seqvec/embeddings/bert_embeddings/embeddings_file_summed.h5',
                           output_path='/mnt/project/bio_embeddings/runs/hannes/combined_embeddings/' + type + '_' + stage + '.h5',
                           type=type)
