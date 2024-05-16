import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from training.trainer import Trainer


def main():
    hparams = "sample_usstof_win.json"
    trainer = Trainer(hparams_file=hparams)
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()


