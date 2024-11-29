import os
import lightning.pytorch as pl
import utils.options as opt

from dataset.hirise_datamodule import HiRISEDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from model import Model


if __name__ == "__main__":
    args = opt.initialize().parse_args()

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=-1,
        monitor="mae",
        mode="max",
        save_last=True
    )

    pl.seed_everything(42, workers=True)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision="16-mixed",
        benchmark=True,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback]
    )

    model = Model(max_depth=700.0)
    datamodule = HiRISEDataModule(data_dir=args.data_dir, batch_size=args.batch_size)

    if not args.test:
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path
        )
    else:  
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path
        )

# nohup python train.py --data_dir /home/super/datasets-nas/HiRISE --batch_size 2 > train.log 2>&1 &
# python train.py --test --data_dir /home/super/datasets-nas/HiRISE --ckpt_path /home/super/mgatti/MarsDEM/pretrained/best_model.ckpt --batch_size 8
