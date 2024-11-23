import torch
import torch.nn as nn
import numpy as np

class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, vector_dim):
        super(BidirectionalLSTM, self).__init__()

        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

    def forward(self, inputs, batch_size=1):
        c0 = torch.rand(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size, requires_grad=False).cuda()
        h0 = torch.rand(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size, requires_grad=False).cuda()
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output, hn, cn

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.task_feat=='output_max':
            self.register_buffer("queue", torch.randn(args.K, args.D*2))
        elif args.task_feat=='hn_mean':
            self.register_buffer("queue", torch.randn(args.K, args.D))
        if self.args.pool_before_lstm:
            self.lstm = BidirectionalLSTM(layer_sizes=[args.D], vector_dim = args.dim_model)
        else:
            self.lstm = BidirectionalLSTM(layer_sizes=[args.D], vector_dim = args.dim_model*args.spatial_dim*args.spatial_dim)
        self.K = args.K
        self.m = args.M
        self.T = args.T
        
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # classes of task in quene
        self.classes = np.ones((self.K, args.way), dtype=int)*1000

        if args.mem_init == 'random':
            memory_tensor = torch.randn(args.mem_len, args.dim_model*args.spatial_dim*args.spatial_dim)
        elif args.mem_init == 'pre_train':
            memory_tensor = torch.load(args.memory)
            if args.mem_init_pooling == 'max':
                memory_tensor = nn.functional.max_pool2d(memory_tensor.reshape(args.mem_len, -1, 5, 5), kernel_size=5)
            if args.mem_init_pooling == 'mean':
                memory_tensor = nn.functional.avg_pool2d(memory_tensor.reshape(args.mem_len, -1, 5, 5), kernel_size=5)
        if args.mem_2d_norm:
            memory_tensor = memory_tensor.view(args.mem_len, args.dim_model, args.spatial_dim, args.spatial_dim)
            memory_tensor = nn.functional.normalize(memory_tensor, dim=1)
            memory_tensor = memory_tensor.view(args.mem_len, args.dim_model*args.spatial_dim*args.spatial_dim)
        if args.mem_grad:
            self.memory = nn.Parameter(memory_tensor.reshape(args.mem_len, -1))
        else:
            self.register_buffer("memory", memory_tensor.reshape(args.mem_len, -1))

        if args.mem_share:
            self.share_memory = nn.Parameter(torch.randn(args.mem_share, args.dim_model*args.spatial_dim*args.spatial_dim))

        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet(pooling=args.pooling)
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = args.dim_model
            from model.networks.res12 import ResNet
            self.encoder = ResNet(drop_rate=args.drop_rate, out_dim=hdim, pooling=args.pooling)
        else:
            raise ValueError('')
        self.hdim = hdim
                       
    @torch.no_grad()
    def _dequeue_and_enqueue(self, key, key_cls):
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr] = key
        self.classes[ptr] = key_cls
        # move pointer
        ptr = (ptr + 1) % self.K  
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_memory(self):
        self.eval()
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        from torch.utils.data import DataLoader
        trainset = Dataset('train', self.args, augment=False, return_id=True, return_simclr=self.args.return_simclr)
        train_loader = DataLoader(dataset=trainset,
                                    num_workers=self.args.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
        from tqdm import tqdm
        embs = torch.zeros(38400, 1600)
        new_protos = torch.zeros(64, 1600)
        for i, batch in tqdm(enumerate(train_loader, 1)):
            data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
            embs[i-1] = self.encoder(data).reshape(-1).detach().cpu()

        for i in range(64):
            new_protos[i] = embs[i*600:(i+1)*600].mean(dim=0)

        self.memory = new_protos.cuda()

    def split_instances(self, actual_shot):
        args = self.args

        mix_train_indices = np.arange(args.way * args.shot)
        mix_train_indices = mix_train_indices.reshape(args.way, args.shot)
        selected_indices = mix_train_indices[:, :actual_shot]

    
        if self.training:
            return  (torch.Tensor(selected_indices).long().reshape(1, actual_shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))

    
    def forward(self, x, ids=None, simclr_images=None, key_cls=None, test=False):
        
        # split support query set for few-shot data
        import random
        

        if self.training:
            # print(actual_shot)
            # print(support_idx)
            actual_shot = random.randint(1,5)
        
            support_idx, query_idx = self.split_instances(actual_shot=5)

            # if actual_shot == 5:
            #     percent = [0.0, 0.2, 0.4, 0.6]
            # elif actual_shot == 4:
            #     percent = [0.0, 0.2, 0.4]
            # elif actual_shot == 3:
            #     percent = [0.0, 0.2]
            # else:
            #     percent = [0.0]
            # percent = [0.0, 0.2, 0.4]
            percent = [0.0]
            from model.noisy.utils import prepare_data
            x = prepare_data(self.args, x, 'sym_swap', percent)
            # percent = random.choice(percent)
            print(percent)
            
            # percent = [0.0, 0.2]
            # percent = 0.6
            # if percent > 0:
            #     from model.noisy.utils import prepare_data
            #     # print(x.shape)
            #     x = prepare_data(self.args, x, 'sym_swap', percent)
            #     # print(x.shape)
        else:
            support_idx, query_idx = self.split_instances(actual_shot=5)


        # feature extraction
        x = x.squeeze(0)
        instance_embs = self.encoder(x)

        simclr_embs = None
        if simclr_images is not None:
            n_embs, n_views, n_ch, spatial, _ = simclr_images.shape
            simclr_images = simclr_images.reshape(-1, n_ch, spatial, spatial)
            simclr_embs = self.encoder(simclr_images)
            spatial_out = simclr_embs.shape[-1]
            simclr_embs = simclr_embs.reshape(n_embs, n_views, self.hdim, spatial_out, spatial_out)

        if self.training:
            logits, logits_simclr, metrics, sims, pure_index, logits_blstm = self._forward(instance_embs, 
                support_idx, query_idx, ids=ids, simclr_embs=simclr_embs, key_cls=key_cls, actual_shot = actual_shot)
            return logits, logits_simclr, metrics, sims, pure_index, logits_blstm
        else:
            if test:
                origin_proto, proto, query = self._forward(instance_embs, support_idx, query_idx, ids=ids, key_cls=key_cls, test=test)
                return origin_proto, proto, query
            logits, logits_blstm = self._forward(instance_embs, support_idx, query_idx, ids=ids, key_cls=key_cls)
            return logits, logits_blstm

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')