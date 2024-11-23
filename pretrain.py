import argparse
import os
import os.path as osp
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.models.classifier import Classifier
from model.dataloader.samplers import CategoriesSampler
from model.utils import pprint, set_gpu, Averager, Timer, count_acc
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    # print('a,b', [a.shape, b.shape])
    logits = -((a - b)**2).sum(dim=2)
    # print('logits', [logits.shape])
    return logits

def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)

# pre-train model, compute validation acc after 500 epoches
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--dataset', type=str, default='TieredImagenet', choices=['MiniImageNet', 'TieredImagenet', 'CUB'])    
    parser.add_argument('--backbone_class', type=str, default='ConvNet', choices=['ConvNet', 'Res12'])
    parser.add_argument('--schedule', type=str, default="75, 150, 300", help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--query', type=int, default=15)    
    parser.add_argument('--resume', type=bool, default=False)
    args = parser.parse_args()
    args.im_size = 1288
    pprint(vars(args))
    
    args.schedule = [int(i) for i in args.schedule.split(',')]
    print('schedule ', args.schedule)
    # asd
    save_path1 = '-'.join([args.dataset, args.backbone_class, 'Pre-random'])
    save_path2 = '_'.join([str(args.lr), str(args.gamma), str(args.schedule)])
    args.save_path = osp.join(save_path1, save_path2)
    if not osp.exists(save_path1):
        os.mkdir(save_path1)
    ensure_path(args.save_path)

    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImagenet':
        from model.dataloader.tiered_imagenet_og import tieredImageNet_og as Dataset    
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args, augment=True)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    args.num_class = trainset.num_class
    print(args.num_class)
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label, 100, 5, 16) # test on 16-way 1-shot
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=8, pin_memory=True)
    args.way = 5
    args.shot = 1
    
    # construct model
    model = Classifier(args)
    if 'Conv' in  args.backbone_class:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif 'Res' in args.backbone_class:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Encoder')    
    criterion = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.ngpu  > 1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(range(args.ngpu)))
        
        model = model.cuda()
        criterion = criterion.cuda()
    
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    def save_checkpoint(is_best, filename='checkpoint.pth.tar'):
        state = {'epoch': epoch + 1,
                 'args': args,
                 'state_dict': model.state_dict(),
                 'trlog': trlog,
                 'val_acc_dist': trlog['max_acc_dist'],
                 'val_acc_sim': trlog['max_acc_sim'],
                 'optimizer' : optimizer.state_dict(),
                 'global_count': global_count}
        
        torch.save(state, osp.join(args.save_path, filename))
        if is_best:
            shutil.copyfile(osp.join(args.save_path, filename), osp.join(args.save_path, 'model_best.pth.tar'))
    
    if args.resume == True:
        # load checkpoint
        state = torch.load(osp.join(args.save_path, 'model_best.pth.tar'))
        init_epoch = state['epoch']
        resumed_state = state['state_dict']
        # resumed_state = {'module.'+k:v for k,v in resumed_state.items()}
        model.load_state_dict(resumed_state)
        trlog = state['trlog']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        global_count = state['global_count']
    else:
        init_epoch = 1
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss_dist'] = []
        trlog['val_loss_sim'] = []
        trlog['train_acc'] = []
        trlog['val_acc_sim'] = []
        trlog['val_acc_dist'] = []
        trlog['max_acc_dist'] = 0.0
        trlog['max_acc_dist_epoch'] = 0
        trlog['max_acc_sim'] = 0.0
        trlog['max_acc_sim_epoch'] = 0        
        initial_lr = args.lr
        global_count = 0

    timer = Timer()
    writer = SummaryWriter(logdir=args.save_path)
    for epoch in range(init_epoch, args.max_epoch + 1):
        # refine the step-size
        if epoch in args.schedule:
            initial_lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        
        model.train()
        tl = Averager()
        ta = Averager()
        print('len of train loader', len(train_loader))
        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, label = [_.cuda() for _ in batch]
                label = label.type(torch.cuda.LongTensor)
            else:
                data, label = batch
                label = label.type(torch.LongTensor)
            # print('HEREEEEEE')
            # print('data', [data.shape, data.min(), data.max()])
            logits = model(data)
            loss = criterion(logits, label)
            acc = count_acc(logits, label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            if (i-1) % 100 == 0:
                print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        print('train acc: ', ta)

        # do not do validation in first 500 epoches
        # a = 1
        if epoch > 100 or (epoch-1) % 5 == 0:
        # if a == 1:
            model.eval()
            vl_dist = Averager()
            va_dist = Averager()
            vl_sim = Averager()
            va_sim = Averager()            
            print('[Dist] best epoch {}, current best val acc={:.4f}'.format(trlog['max_acc_dist_epoch'], trlog['max_acc_dist']))
            # print('[Sim] best epoch {}, current best val acc={:.4f}'.format(trlog['max_acc_sim_epoch'], trlog['max_acc_sim']))
            # test performance with Few-Shot
            label = torch.arange(5).repeat(args.query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)        
            with torch.no_grad():
                for i, batch in tqdm(enumerate(test_loader, 1)):
                    if torch.cuda.is_available():
                        data, dummy_label = [_.cuda() for _ in batch]
                    else:
                        data, _ = batch
                    
                    # print(data.shape)
                    data_shot, data_query = data[:5], data[5:] # 16-way test

                    # print('putting al zero in data query 5')
                    # data_query[5,:,:,:] = torch.zeros(3, 84, 84)

                    logits_dist, logits_sim = model.forward_proto(data_shot, data_query, testset.num_class)
                    # asd
                    # label = dummy_label[valset.num_class:]

                    # non sorted does not work!! WHyyyyyyyyyyyyyy!

                    # print('here', data_shot.shape)
                    # print('here', data_query.shape)
                    # print('here', data_shot.shape)
                    # print('here', data_query.shape)
                    # print('data', [data.shape, data.min(), data.max()])
                    # np.save('./img.npy', data.cpu().numpy())
                    # print('labels', label)
                    # print('dummy_label', dummy_label)
                    # # print(label)
                    # asd


                    loss_dist = F.cross_entropy(logits_dist, label)
                    acc_dist = count_acc(logits_dist, label)
                    loss_sim = F.cross_entropy(logits_sim, label)
                    acc_sim = count_acc(logits_sim, label)                    
                    vl_dist.add(loss_dist.item())
                    va_dist.add(acc_dist)
                    vl_sim.add(loss_sim.item())
                    va_sim.add(acc_sim)                    

            vl_dist = vl_dist.item()
            va_dist = va_dist.item()
            vl_sim = vl_sim.item()
            va_sim = va_sim.item()            
            writer.add_scalar('data/val_loss_dist', float(vl_dist), epoch)
            writer.add_scalar('data/val_acc_dist', float(va_dist), epoch)     
            writer.add_scalar('data/val_loss_sim', float(vl_sim), epoch)
            writer.add_scalar('data/val_acc_sim', float(va_sim), epoch)               
            print('epoch {}, val, loss_dist={:.4f} acc_dist={:.4f} '.format(epoch, vl_dist, va_dist))
    
            if va_dist > trlog['max_acc_dist']:
                trlog['max_acc_dist'] = va_dist
                trlog['max_acc_dist_epoch'] = epoch
                save_model('max_acc_dist')
                save_checkpoint(True)
                
            if va_sim > trlog['max_acc_sim']:
                trlog['max_acc_sim'] = va_sim
                trlog['max_acc_sim_epoch'] = epoch
                save_model('max_acc_sim')
                save_checkpoint(True)            
    
            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss_dist'].append(vl_dist)
            trlog['val_acc_dist'].append(va_dist)
            trlog['val_loss_sim'].append(vl_sim)
            trlog['val_acc_sim'].append(va_sim)            
            save_model('epoch-last')
    
            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()
    
    
    import pdb
    pdb.set_trace()