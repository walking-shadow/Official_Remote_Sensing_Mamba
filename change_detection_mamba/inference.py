import sys
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
import logging
from utils.path_hyperparameter import ph
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from rs_mamba_cd.rs_mamba_cd import RSM_CD
from tqdm import tqdm


def train_net(dataset_name, load_checkpoint=True):
    # 1. Create dataset

    test_dataset = BasicDataset(t1_images_dir=f'./{dataset_name}/test/t1/',
                                t2_images_dir=f'./{dataset_name}/test/t2/',
                                labels_dir=f'./{dataset_name}/test/label/',
                                train=False)
    # 2. Create data loaders
    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True
                       )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 3. Initialize logging
    logging.basicConfig(level=logging.INFO)

    # 4. Set up device, model, metric calculator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Using device {device}')
    net = RSM_CD(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank, \
               ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version, patchembed_version=ph.patchembed_version)
    net.to(device=device)

    assert ph.load, 'Loading model error, checkpoint ph.load'
    load_model = torch.load(ph.load, map_location=device)
    if load_checkpoint:
        net.load_state_dict(load_model['net'])
    else:
        net.load_state_dict(load_model)
    logging.info(f'Model loaded from {ph.load}')
    torch.save(net.state_dict(), f'{dataset_name}_best_model.pth')

    metric_collection = MetricCollection({
        'accuracy': Accuracy().to(device=device),
        'precision': Precision().to(device=device),
        'recall': Recall().to(device=device),
        'f1score': F1Score().to(device=device)
    })  # metrics calculator

    net.eval()
    logging.info('SET model mode to test!')

    with torch.no_grad():
        for batch_img1, batch_img2, labels, name in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.float().to(device)

            cd_preds = net(batch_img1, batch_img2, log=True, img_name=name)
            cd_preds = torch.sigmoid(cd_preds)

            # Calculate and log other batch metrics
            cd_preds = cd_preds.float()
            labels = labels.int().unsqueeze(1)
            metric_collection.update(cd_preds, labels)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        test_metrics = metric_collection.compute()
        print(f"Metrics on all data: {test_metrics}")
        metric_collection.reset()

    print('over')


if __name__ == '__main__':

    try:
        train_net(dataset_name='your_dataset_name')
    except KeyboardInterrupt:
        logging.info('Error')
        sys.exit(0)
