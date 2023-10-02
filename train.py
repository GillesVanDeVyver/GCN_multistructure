### Import external libraries
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import socket

### Import internal libraries
import CONST
from engine.loops import sample_dataset, train_vanilla, validate
from engine.checkpoints import save_model
from evaluation.EchonetEvaluator import EchonetEvaluator
from models import load_model
from losses import load_loss
from datasets import load_dataset
from transforms import load_transform
from optimizers import load_optimizer
from utils.utils_files import better_hparams
from config.defaults import cfg_costum_setup, default_argument_parser, get_run_id, convert_to_dict, \
    create_tensorboard_run_dict

logs_dir = CONST.STORAGE_DIR


def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if 'it/s]' not in line)
    return '\n'.join(lines)


# ===========================
# ===========================
# Main
# ===========================
# ===========================

def train(cfg):
    print("Train Config:\n", cfg)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    hostname = socket.getfqdn()
    run_id = get_run_id()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        is_gpu = True
    else:
        is_gpu = False
    print(device)
    config_dict = convert_to_dict(cfg, [])

    # ----- Load data -----:
    # Input image transformer:
    if cfg.AUG.PROB > 0:
        insize = cfg.TRAIN.INPUT_SIZE
        input_transform = load_transform(augmentation_type=cfg.AUG.METHOD,
                                         augmentation_probability=cfg.AUG.PROB,
                                         input_size=insize, num_frames=cfg.NUM_FRAMES)
    else:
        input_transform = None
    # dataset object:
    if cfg.TRAIN.NUM_KPTS_MULTISTRUCTURE is not None:
        num_kpts = cfg.TRAIN.NUM_KPTS_MULTISTRUCTURE
    else:
        num_kpts = cfg.TRAIN.NUM_KPTS
    ds = load_dataset(ds_name=cfg.TRAIN.DATASET, input_transform=input_transform, input_size=cfg.TRAIN.INPUT_SIZE,
                      num_frames=cfg.NUM_FRAMES,
                      num_kpts=num_kpts)
    # data loaders:
    trainloader, valloader, _ = sample_dataset(trainset=ds.trainset,
                                               valset=ds.valset,
                                               testset=None,
                                               overfit=cfg.TRAIN.OVERFIT,
                                               batch_size=cfg.TRAIN.BATCH_SIZE,
                                               num_workers=cfg.DATA_LOADER.NUM_WORKERS)

    # ----- Load model -----:
    if cfg.TRAIN.LOAD_OPTIMIZER:
        model, optimizer_state_dict = load_model(cfg, is_gpu=is_gpu, load_optimizer_dict=True)
    else:
        model = load_model(cfg, is_gpu=is_gpu)

    total_nb_params = sum(p.numel() for p in model.parameters())
    print('total number of params: ' + str(total_nb_params))
    model.to(device)
    optimizer = load_optimizer(method_name=cfg.SOLVER.OPTIMIZER, parameters=model.parameters(),
                               learningrate=cfg.SOLVER.BASE_LR)
    if cfg.TRAIN.LOAD_OPTIMIZER:
        optimizer.load_state_dict(optimizer_state_dict)

    print('training model {}..'.format(model.__class__.__name__))
    # if resume training:
    if (cfg.TRAIN.WEIGHTS is not None) and (os.path.exists(cfg.TRAIN.WEIGHTS)):
        print("loading weights {}..".format(cfg.TRAIN.WEIGHTS))
        checkpoint = torch.load(cfg.TRAIN.WEIGHTS)
        model.load_state_dict(checkpoint['model_state_dict'])
    # Set training parameters:
    class_weights = torch.Tensor([1] * (1 + cfg.NUM_FRAMES));
    class_weights[0] = 0.1;
    class_weights /= len(class_weights)
    class_weights = class_weights.to(device)
    if len(cfg.MODEL.LOSS_FUNC) == 1:
        loss = cfg.MODEL.LOSS_FUNC[0]
    else:
        loss = cfg.MODEL.LOSS_FUNC  # workaround for handling multiple losses in cfg
    criterion = load_loss(loss, device=device, class_weights=class_weights)
    # Load optimizer:

    # ----- Setup logger -----:
    best_val_loss = np.inf
    best_val_metric = {"kpts": np.inf, "ef": np.inf, "sd": np.inf, 'distances': np.inf}
    log_folder = os.path.join(logs_dir, 'logs', cfg.TRAIN.DATASET,
                              cfg.MODEL.NAME, str(cfg.MODEL.BACKBONE), run_id)

    writer = SummaryWriter(log_dir=log_folder)
    metric_dict = {'BestVal/kptsErr': 1, 'BestVal/efErr': 1, 'BestVal/sdErr': 1}
    run_dict = create_tensorboard_run_dict(cfg)
    run_dict["hostname"] = hostname

    sei = better_hparams(writer, hparam_dict=run_dict, metric_dict=metric_dict)
    basename = "{}_{}".format(cfg.TRAIN.DATASET, cfg.MODEL.NAME)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    with open(os.path.join(log_folder, "train_config.yaml"), "w") as f:
        f.write(cfg.dump())  # save config to file

    # ----- Train & Evaluate: -----
    print("Training in batches of size {}..".format(cfg.TRAIN.BATCH_SIZE))
    print('Training on machine name {}..'.format(hostname))
    print("Using data augmentation type {} for {:.2f}% of the input data".format(cfg.AUG.METHOD, 100 * cfg.AUG.PROB))
    evaluator = EchonetEvaluator(dataset=ds.valset, output_dir=None, verbose=False)
    with tqdm(total=cfg.TRAIN.EPOCHS) as pbar_main:
        for epoch in range(1, cfg.TRAIN.EPOCHS + 1):
            pbar_main.update()

            train_losses = train_vanilla(epoch=epoch,
                                         loader=trainloader,
                                         optimizer=optimizer,
                                         model=model,
                                         device=device,
                                         criterion=criterion,
                                         prossesID=run_id)
            train_loss = train_losses["main"].avg
            writer.add_scalar('Loss/Train', train_loss, epoch)

            # eval:
            if cfg.TRAIN.SAVE_INTERVAL is not None and epoch % cfg.TRAIN.SAVE_INTERVAL == 0:
                interval_save = True
            else:
                interval_save = False

            if epoch % cfg.TRAIN.EVAL_INTERVAL == 0 or interval_save:
                val_losses, val_outputs, val_inputs = validate(mode='validation',
                                                               epoch=epoch,
                                                               loader=valloader,
                                                               model=model,
                                                               device=device,
                                                               criterion=criterion,
                                                               prossesID=run_id)
                val_loss = val_losses["main"].avg
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                for task in ['ef', 'sd', 'kpts', 'distances']:
                    if task in val_losses:
                        writer.add_scalar("Loss/{}_Validation".format(task), val_losses[task].avg, epoch)

                evaluator.process(val_inputs, val_outputs)
                eval_metrics = evaluator.evaluate()
                for task in ['ef', 'sd', 'kpts', 'distances']:
                    if task in evaluator.get_tasks():
                        writer.add_scalar("Val/{}ERR".format(task, task), eval_metrics[task], epoch)

                if val_loss < best_val_loss:
                    filename = os.path.join(log_folder, 'weights_{}_best_loss.pth'.format(basename))
                    best_val_loss = val_loss
                    save_model(filename, epoch, model, cfg, train_loss, val_loss, best_val_metric, hostname,
                               optimizer)
                    print("Saved at loss {:.5f}\n".format(val_loss))
                    writer.add_scalar('BestVal/Loss', best_val_loss, epoch)

                if interval_save:
                    filename = os.path.join(log_folder, 'weights_{}_ep_{}.pth'.format(basename, epoch))
                    save_model(filename, epoch, model, cfg, train_loss, val_loss, best_val_metric, hostname,
                               optimizer)

                # Update best val metric:
                for task in ['ef', 'sd', 'kpts', 'distances']:
                    if task in eval_metrics and eval_metrics[task] < best_val_metric[task]:
                        filename = os.path.join(log_folder, 'weights_{}_best_{}Err.pth'.format(basename, task))
                        best_val_metric[task] = eval_metrics[task]
                        writer.add_scalar("BestVal/{}Err".format(task), best_val_metric[task], epoch)
                        if task in ['ef', 'kpts', 'distances']:
                            save_model(filename, epoch, model, cfg, train_loss, val_loss, best_val_metric, hostname,
                                       optimizer)
                            print("Saved at val loss {:.5f}, {} error {:.5f}%\n".format(val_loss, task,
                                                                                        eval_metrics[task]))
                writer.flush()
    # Save & Close:
    print('Finished Training')
    filename = os.path.join(log_folder, 'weights_{}_ep_{}.pth'.format(basename, epoch))
    save_model(filename, epoch, model, cfg, train_loss, val_loss, best_val_metric, hostname,
               optimizer)
    writer.file_writer.add_summary(sei)
    writer.close()  # close tensorboard


if __name__ == '__main__':
    args = default_argument_parser()
    cfg = cfg_costum_setup(args)

    train(cfg)
