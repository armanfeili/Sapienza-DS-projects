from .config import *
import PIL.Image
import torchvision.transforms as T


def get_preprocessing(config: VisualConfig, is_training=True):

    transform = []

    if is_training:
        transform.append(T.RandomRotation(15))
        transform.append(
            T.RandomResizedCrop(size=config.train_input_size, scale=(0.9, 1))
        )

        transform.append(
            T.ColorJitter(
                config.aug_color_jitter_b,
                config.aug_color_jitter_c,
                config.aug_color_jitter_s,
                0.0,
            )
        )
        transform.append(T.RandomHorizontalFlip())
    else:
        transform.append(T.Resize(config.test_input_size))

    transform.append(T.ToTensor())
    transform.append(T.Normalize(config.norm_mean, config.norm_std))

    return T.Compose(transform)
