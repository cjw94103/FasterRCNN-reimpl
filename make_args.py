from utils import load_json_file

class Args:
    def __init__(self, config_path):
        self.config = load_json_file(config_path)
        self.data_path = self.config['data_path']
        self.num_workers = self.config['num_workers']
        self.batch_size = self.config['batch_size']
        
        self.backbone = self.config['backbone']
        self.anchor_sizes = self.config['anchor_sizes']
        self.anchor_ratio = self.config['anchor_ratio']
        self.pooler_output_size = self.config['pooler_output_size']
        self.pooler_sampling_ratio = self.config['pooler_sampling_ratio']
        
        self.epochs = self.config['epochs']
        self.model_save_path = self.config['model_save_path']
        self.save_per_epochs = self.config['save_per_epochs']
        self.monitor = self.config['monitor']
        
        self.multi_gpu_flag = self.config['multi_gpu_flag']
        self.port_num = self.config['port_num']

        self.lr = self.config['lr']
        self.weight_decay = self.config['weight_decay']
        self.momentum = self.config['momentum']
        