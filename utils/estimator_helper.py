from utils.metrics import Estimator

def get_estimator(cfg):
    return Estimator(cfg.train.metrics, cfg.data.num_classes, cfg.train.criterion)
