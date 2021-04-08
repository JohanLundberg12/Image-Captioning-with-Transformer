class Config(object):
    def __init__(self, lang, embedding_type, samples):
        
        # Training Type
        self.batch_size = 32
        self.embedding_type = embedding_type
        self.epochs = 30
        self.lang = lang
        self.samples = samples
        self.seed = 2

        # Data
        self.checkpoint_path = './checkpoints'
        self.image_path = 'data/Flickr8k_Dataset/'

        #Transformer
        self.col_size = 8
        self.dff = 2048
        self.d_model = 768
        self.num_heads = 8
        self.num_layers = 4
        self.rate = 0.1
        self.row_size = 8

