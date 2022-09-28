
from .dataloader import NYUDataset
from config import Path
from .semantic_kitti import SemanticKittiDataset
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):

    if args.dataset:
        base_dirs = Path.db_root_dir(args.dataset)

        print('Training data:{}'.format(base_dirs['train']))
        train_loader = DataLoader(
            dataset=SemanticKittiDataset(base_dirs['train'], 'train' ,subfix='npz',istest=False),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )

        print('Validate data:{}'.format(base_dirs['val']))
        val_loader = DataLoader(
            dataset=SemanticKittiDataset(base_dirs['val'], 'val',subfix='npz', istest=True),
            batch_size=args.batch_size,  # 1 * torch.cuda.device_count(), 1 for each GPU
            shuffle=False,
            num_workers=args.workers  # 1 * torch.cuda.device_count()
        )

        return train_loader, val_loader
