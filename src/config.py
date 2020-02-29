class Config():
    def __init__(self, home=True):
        self.seed = 71
        self.batch_size = 128
        self.accum_time = 1
        self.train_csv = 'input/train.csv'
        self.train_images = 'input/train_128x128_orig.npy'
        self.test_csv = 'input/test_orig.csv'
        self.test_images = 'input/test_128x128_orig.npy'
        self.npy = True
        
        self.device_name = 'cuda:0'
        self.weighted_sample = True
        self.image_size = (137, 236)

        self.stratify = "random"
        self.n_splits = 5
        self.fold = 0
        self.num_epoch = 150
        self.period = 16

        self.arch = "se_resnet50"
        self.gr_size = 168
        self.vd_size = 11
        self.cd_size = 7

        self.mask = False
        self.mask_size = (40, 64)        

        self.mixup = True
        self.mixup_prob = 0.5
        self.alpha = 0.7
        self.beta = 0.7

        self.augmix = False
        self.augmix_prob = 1

        self.init_lr = 2e-4
        self.eta_min = 1e-6
        self.num_workers = 16 if home else 4
        self.classes_num = 1
    

conf = Config(home=True)

