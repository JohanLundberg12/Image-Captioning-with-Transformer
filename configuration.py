class Config(object):
    def __init__(self, lang, embedding_type):
        
        # Training Type
        self.lang = lang
        self.embedding_type = embedding_type

        # Data
        self.image_path = 'data/Flickr8k_Dataset/'
        self.samples = 64
        self.checkpoint_path = './checkpoints'


        #Basic
        self.seed = 42
        self.epochs = 1

        #Transformer
        self.num_layers = 4
        self.d_model = 768
        self.dff = 2048
        self.num_heads = 8
        self.row_size = 8
        self.col_size = 8
        self.rate = 0.1

