
'''
train on 4 NVIDIA 1080Ti GPUs
'''
class config_func():
    def __init__(self, source_domain, target_domain):

        self.lr_1stStage = 1e-4
        if source_domain == 'D1':
            self.lr_1stStage = 2e-4

        elif source_domain == 'D2':
            self.lr_1stStage = 9e-5

