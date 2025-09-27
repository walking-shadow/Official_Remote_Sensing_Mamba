import sys, os
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
import logging
from utils.path_hyperparameter import ph
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from rs_mamba_ss import RSM_SS
from tqdm import tqdm
import PIL.Image
import PIL
import numpy as np
import albumentations as A

if ph.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(ph.gpu_id)

def save_prediction(prediction: torch.Tensor, filepath: str):
    tensor_softmax = torch.nn.Softmax(dim=0)(prediction)
    tensor_argmax = torch.argmax(tensor_softmax, dim=0)
    np_argmax = tensor_argmax.numpy().astype(np.uint8)
    np_argmax[np_argmax == 1] = 85
    np_argmax[np_argmax == 2] = 170
    
    img_pil = PIL.Image.fromarray(np_argmax,mode='L')
    img_pil.save(f'{filepath}.png')

def infer(images_folder, output_folder):
    """
    Function to infer a folder of images and save the predictions.
    
    Args:
        images_folder (str): Path to the folder containing images.
        output_folder (str): Path to the folder where predictions will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RSM_SS(num_classes=ph.num_classes, dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank, \
               ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio)
    net.to(device=device)

    load_model = torch.load(ph.load, map_location=device)
    net.load_state_dict(load_model)
    net.eval()

    for img_name in os.listdir(images_folder):

        img_path = os.path.join(images_folder, img_name)
        img = PIL.Image.open(img_path)
        img = np.array(img).astype(np.uint8)
        normalized = A.Normalize()(image=img)['image']
        normalized = normalized.transpose(2, 0, 1)  # Convert to CxHxW format
        normalized = np.expand_dims(normalized, axis=0)  # Add batch dimension
        img_tensor = torch.tensor(normalized).float().to(device)
        #print(f"img_tensor memory size: {img_tensor.element_size() * img_tensor.nelement() / 1024:.2f} KB")
        with torch.no_grad():
            pred = net(img_tensor)
        save_prediction(np.squeeze(pred.detach().cpu()), os.path.join(output_folder, img_name.split('.')[0]))
        logging.info(f'Prediction saved for {img_name} in {output_folder}')
        
        del img_tensor, pred  # clear batch variables from memory
    
    del net, load_model  # clear model variables from memory
    torch.cuda.empty_cache()  # clear GPU memory
    logging.info('Inference completed and memory cleared.')


def infer_test_set(dataset_name, load_checkpoint=True, save_prediction_img=False):
    # 1. Create dataset

    test_dataset = BasicDataset(images_dir=f'{ph.root_dir}/{dataset_name}/test/image/',
                                labels_dir=f'{ph.root_dir}/{dataset_name}/test/label/',
                                train=False)
    # 2. Create data loaders
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
    net = RSM_SS(num_classes=ph.num_classes, dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank, \
               ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio)
    net.to(device=device)

    assert ph.load, 'Loading model error, checkpoint ph.load'
    load_model = torch.load(ph.load, map_location=device)
    net.load_state_dict(load_model)
    logging.info(f'Model loaded from {ph.load}')

    metric_collection = MetricCollection({
        'accuracy': Accuracy(ignore_index=None, num_classes=ph.num_classes, multiclass=True, mdmc_average='global').to(device=device),
        'precision': Precision(ignore_index=None, num_classes=ph.num_classes, multiclass=True, mdmc_average='global').to(device=device),
        'recall': Recall(ignore_index=None, num_classes=ph.num_classes, multiclass=True, mdmc_average='global').to(device=device),
        'f1score': F1Score(ignore_index=None, num_classes=ph.num_classes, multiclass=True, mdmc_average='global').to(device=device)
    })  # metrics calculator

    net.eval()
    logging.info('SET model mode to test!')

    with torch.no_grad():
        for batch_img1, labels, names in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            labels = labels.float().to(device)

            ss_preds = net(batch_img1)
            
            if save_prediction_img:
                save_pred_directory = f"./{os.path.basename(ph.root_dir)}_{os.path.basename(ph.load).replace('.pth','')}"
                if not os.path.exists(save_pred_directory): os.makedirs(save_pred_directory)

                for pred_img, name in zip(ss_preds, names):
                    save_prediction(pred_img.detach().cpu(), os.path.join(save_pred_directory, name))

            # Calculate and log other batch metrics
            metric_collection.update(ss_preds.float(), labels.long())

            # clear batch variables from memory
            del batch_img1, labels

        test_metrics = metric_collection.compute()
        print(f"Metrics on all data: {test_metrics}")
        metric_collection.reset()

    print('over')


if __name__ == '__main__':

    try:
        #infer_test_set(dataset_name=ph.dataset_name, save_prediction_img=True)
        infer(images_folder='/mnt/0_ARCTUS_Projects/18_MEI_SDB2/data/s2_gee_HBE/data_training_osw/good_bad_inputs/Good_sample/',
              output_folder='/home/hurens/Documents/Official_Remote_Sensing_Mamba/simple_inference/output_on_val_set_oswnet')
    except KeyboardInterrupt:
        logging.info('Error')
        sys.exit(0)
