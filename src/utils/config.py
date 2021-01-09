from omegaconf import OmegaConf


def print_cfg(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
