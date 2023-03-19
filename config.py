class Config():
    def __init__(self, args):
        
        self.debug = args.debug # enable/disable debug logs
        self.batch_size = args.batch_size # batch size for train/test
        self.dataset = args.dataset # dataset that we will use
        self.num_inputs = args.num_inputs # number of features
        self.num_hidden = args.num_hidden # number of hidden layers
        self.num_outputs = args.num_outputs # number of labels
        self.num_steps = args.num_steps # number of steps
        self.beta = args.beta
        self.num_epochs = args.num_epochs # number of epochs
        self.plot = args.plot
