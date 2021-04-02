class Config(object):
    def __init__(self, lang, embedding_type, samples):
        
        # Training Type
        self.lang = lang
        self.embedding_type = embedding_type
        self.samples = samples
        self.epochs = 5
        self.seed = 2
        self.batch_size = 128

        # Data
        self.image_path = 'data/Flickr8k_Dataset/'
        self.checkpoint_path = './checkpoints'

        #Transformer
        self.num_layers = 4
        self.d_model = 768
        self.dff = 2048
        self.num_heads = 8
        self.row_size = 8
        self.col_size = 8
        self.rate = 0.1

