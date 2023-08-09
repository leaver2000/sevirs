import argparse
import logging
import sys

from .constants import ImageType
from .core.catalog import Catalog
from .core.datasets import TensorGenerator, TensorLoader
from .models.wx_gpt import GPTConfig


def main(config: GPTConfig, patch_size=48, batch_size=2) -> int:
    logging.basicConfig(level=logging.INFO)
    logging.info(f"config: {config}")
    with TensorLoader(
        TensorGenerator(
            Catalog(
                "/mnt/data/sevir",
                img_types=config.inputs + config.targets,
            ),
            inputs=config.inputs,
            targets=config.targets,
            patch_size=patch_size,
        ),
        batch_size=batch_size,
    ) as tl:
        for i, (x, y) in enumerate(tl):
            print(x.shape, y.shape)
            if i > 2:
                break

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--inputs", nargs="+", choices=list(ImageType), default=[ImageType.VIS])
    parser.add_argument("--targets", nargs="+", choices=list(ImageType), default=[ImageType.VIL])
    print(parser.parse_args())
    sys.exit(main(GPTConfig(**vars(parser.parse_args()))))
