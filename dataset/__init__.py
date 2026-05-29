from torch.utils.data import DataLoader
from dataset.himmelblau.himmelblau_dataset import HimmelblauDataset
from dataset.robot.robot_dataset import RobotDataset
from dataset.rosenbrock.rosenbrock_dataset import RosenbrockDataset
from dataset.levy.levy_dataset import LevyDataset
from dataset.qp_constrained.qp_constrained_dataset import QPConstrainedDataset
from dataset.dataset import BaseDataset

def build_dataloader(
    args,
    logger,
    split="train",
):


    if args.task_name == "himmelblau":
        dataset = HimmelblauDataset(
            split=split,
            args=args,
            logger=logger,
        )
    elif args.task_name == "robot":
        dataset = RobotDataset(
            split=split,
            args=args,
            logger=logger,
        )
    elif args.task_name == "rosenbrock":
        dataset = RosenbrockDataset(
            split=split,
            args=args,
            logger=logger,
        )
    elif args.task_name == "levy":
        dataset = LevyDataset(
            split=split,
            args=args,
            logger=logger,
        )
    elif args.task_name == "qp_constrained":
        dataset = QPConstrainedDataset(
            split=split,
            args=args,
            logger=logger,
        )
    else:
        raise ValueError(f"Task name {args.task_name} is not supported!")
    
    # Create dataloader
    shuffle = True if split == "train" or split == "test_random" else False
    drop_last = False
    
    # Select batch size based on split
    if split == "train":
        batch_size = args.train_batch_size
    elif split == "val":
        batch_size = args.validation_batch_size
    elif split == "test" or split == "test_random":
        batch_size = args.test_batch_size
    else:
        raise ValueError(f"Split {split} is not supported!")
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=args.workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=BaseDataset.collate_fn,
    )
    
    if logger is not None:
        logger.info(f"Dataloader for {split} with {len(dataset)} samples has been built!")
        
    return dataloader
