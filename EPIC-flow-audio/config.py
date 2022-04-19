
'''
train on 4 NVIDIA 1080Ti GPUs
'''
class config_func():
    def __init__(self, source_domain, target_domain):
        self.lr_1stStage = 1e-4

        self.lr_2ndStage = 1e-4
