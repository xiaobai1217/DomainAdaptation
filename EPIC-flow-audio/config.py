
'''
train on 4 NVIDIA 1080Ti GPUs
'''
class config_func():
    def __init__(self, source_domain, target_domain):
        self.emb_dropout = 0.1
        self.depth = 2
        self.dropout=0.15
        self.lr_1stStage = 1e-4
        if source_domain == 'D1':
            self.epoch_num_1st = 8
            self.epoch_num_2nd = 14
            self.iter_num_trans = 1450
            if target_domain =='D3':
                self.iter_num_trans = 800

        elif source_domain == 'D2':
            self.epoch_num_1st = 5
            self.epoch_num_2nd = 6
            if target_domain =='D1':
                self.iter_num_trans = 450
            else:
                self.iter_num_trans = 950
                self.emb_dropout = 0.15
                self.dropout = 0.3
        elif source_domain == 'D3':
            self.epoch_num_1st = 5
            self.epoch_num_2nd = 34
            self.iter_num_trans = 1050
            if target_domain =='D1':
                self.iter_num_trans = 2250
                self.emb_dropout = 0.3
                self.dropout = 0.3
                self.depth = 1

        self.lr_2ndStage = 1e-4
