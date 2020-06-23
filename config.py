class Config:
    def __init__(self):
        self.input_size = 500
        self.batch_size = 16
        self.num_epochs = 50
        self.train_file_path = ''
        self.val_file_path = ''
        self.num_classes = 2
        self.filters = []
        self.kernel_size = 3
        self.use_weights = False
        self.num_channels = 3
        self.save_directory = 'logs'
        self.mirror = True
        self.noise = True
        self.rotate = True
        self.learning_rate = 0.001
