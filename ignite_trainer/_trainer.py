﻿import io
import os
import glob
import json
import time
import tqdm
import signal
import argparse
import numpy as np

import torch
import torch.utils.data

import torchvision as tv

import ignite.engine as ieng
import ignite.metrics as imet
import ignite.handlers as ihan

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Optional

from termcolor import colored

from collections import defaultdict
from collections.abc import Iterable

from ignite_trainer import _utils
from ignite_trainer import _interfaces

from ignite.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, DiskSaver


BATCH_TRAIN = 128
BATCH_TEST = 1024
WORKERS_TRAIN = 0
WORKERS_TEST = 0
EPOCHS = 100
LOG_INTERVAL = 50
SAVED_MODELS_PATH = os.path.join(os.path.expanduser('~'), 'saved_models')


def run(experiment_name: str,
        model_class: str,
        model_args: Dict[str, Any],
        optimizer_class: str,
        optimizer_args: Dict[str, Any],
        dataset_class: str,
        dataset_args: Dict[str, Any],
        batch_train: int,
        batch_test: int,
        workers_train: int,
        workers_test: int,
        transforms: List[Dict[str, Union[str, Dict[str, Any]]]],
        epochs: int,
        log_interval: int,
        saved_models_path: str,
        performance_metrics: Optional = None,
        scheduler_class: Optional[str] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        model_suffix: Optional[str] = None,
        setup_suffix: Optional[str] = None,
        orig_stdout: Optional[io.TextIOBase] = None,
        skip_train_val: bool = False
        ):

    with _utils.tqdm_stdout(orig_stdout) as orig_stdout:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            experiment_name = f'{experiment_name}-x{num_gpus}'

        transforms_train = list()
        transforms_test = list()

        for idx, transform in enumerate(transforms):
            use_train = transform.get('train', True)
            use_test = transform.get('test', True)

            transform = _utils.load_class(transform['class'])(**transform['args'])

            if use_train:
                transforms_train.append(transform)
            if use_test:
                transforms_test.append(transform)

            transforms[idx]['train'] = use_train
            transforms[idx]['test'] = use_test

        transforms_train = tv.transforms.Compose(transforms_train)
        transforms_test = tv.transforms.Compose(transforms_test)

        Dataset: Type = _utils.load_class(dataset_class)
        
        train_loader, eval_loader = _utils.get_data_loaders(
            Dataset,
            dataset_args,
            batch_train,
            batch_test,
            workers_train,
            workers_test,
            transforms_train,
            transforms_test
        )

        Network: Type = _utils.load_class(model_class)
        
        model: _interfaces.AbstractNet = Network(**model_args)

        model = torch.nn.DataParallel(model, device_ids=range(num_gpus))
        model = model.to(device)

        # add only enabled parameters to optimizer's list
        param_groups = [
            {'params': [p for p in model.module.parameters() if p.requires_grad]}
        ]

        Optimizer: Type = _utils.load_class(optimizer_class)
        optimizer: torch.optim.Optimizer = Optimizer(
            param_groups,
            **{**optimizer_args, **{'lr': optimizer_args['lr'] * num_gpus}}
        )

        if scheduler_class is not None:
            Scheduler: Type = _utils.load_class(scheduler_class)

            if scheduler_args is None:
                scheduler_args = dict()

            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = Scheduler(optimizer, **scheduler_args)
        else:
            scheduler = None

        model_short_name = ''.join([c for c in Network.__name__ if c == c.upper()])
        model_name = '{}{}'.format(
            model_short_name,
            '-{}'.format(model_suffix) if model_suffix is not None else ''
        )

        prog_bar_epochs = tqdm.tqdm(total=epochs, desc='Epochs', file=orig_stdout, dynamic_ncols=True, unit='epoch')
        prog_bar_iters = tqdm.tqdm(desc='Batches', file=orig_stdout, dynamic_ncols=True)

        num_params_total = sum(p.numel() for p in model.parameters())
        num_params_train = sum(p.numel() for grp in optimizer.param_groups for p in grp['params'])

        params_total_label = ''
        params_train_label = ''
        if num_params_total > 1e6:
            num_params_total /= 1e6
            params_total_label = 'M'
        elif num_params_total > 1e3:
            num_params_total /= 1e3
            params_total_label = 'k'

        if num_params_train > 1e6:
            num_params_train /= 1e6
            params_train_label = 'M'
        elif num_params_train > 1e3:
            num_params_train /= 1e3
            params_train_label = 'k'

        tqdm.tqdm.write(f'\n{Network.__name__}\n')
        tqdm.tqdm.write('Total number of parameters: {:.2f}{}'.format(num_params_total, params_total_label))
        tqdm.tqdm.write('Number of trainable parameters: {:.2f}{}'.format(num_params_train, params_train_label))

        def training_step(engine: ieng.Engine, batch) -> torch.Tensor:
            model.train()
            model.epoch = engine.state.epoch
            model.batch_idx = (engine.state.iteration - 1) % len(train_loader)
            model.num_batches = len(train_loader)

            optimizer.zero_grad()

            loss = Network.training_step_imp(model, batch, device)

            if loss.ndim > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step(None)

            return loss.item()

        def eval_step(engine: ieng.Engine, batch) -> _interfaces.TensorPair:
            model.eval()
            
            default_samples_path = f"{saved_models_path}/samples"
            is_infer_samples = False
            if isinstance(model_args['infer_samples'],str):
                default_samples_path = model_args['infer_samples']
                is_infer_samples = True
            elif isinstance(model_args['infer_samples'],bool):
                is_infer_samples = model_args['infer_samples']
            
            if is_infer_samples and ((engine.state.iteration - 1) % len(train_loader)) == 0:
                save_folder = default_samples_path
                os.makedirs(save_folder, exist_ok=True)
                Network.inference(model, batch, device, save_folder)
                return Network.eval_step_imp(model, batch, device)
            else:
                return Network.eval_step_imp(model, batch, device)
            
        trainer = ieng.Engine(training_step)
        validator_train = ieng.Engine(eval_step)
        validator_eval = ieng.Engine(eval_step)

        #Metrics
        # checkpoint_metrics = list()
        for metric_detail in performance_metrics:
            output_transform = (lambda output: list([output[key] for key in metric_detail['args']['output_transform']]))
            metric_obj: imet.Metric = _utils.load_class(metric_detail['class'])(output_transform=output_transform, device=device)
            metric_label = metric_detail.get('label', 'default_label')
            metric_use_for = metric_detail.get('use_for', [])
            # metric_save_checkpoint = metric_detail.get('save_checkpoint', False)
            
            for use_for in metric_use_for:
                if use_for == "val" and not skip_train_val:
                    metric_obj.attach(validator_train, metric_label)
                elif use_for == "test":
                    metric_obj.attach(validator_eval, metric_label) 
            # if metric_save_checkpoint:
            #     checkpoint_metrics.append(metric_label)

        # Create a logger
        tb_logger = TensorboardLogger(log_dir=f"{saved_models_path}/tb_logs")
        # Attach the logger to the trainer to log training loss at each iteration
        tb_logger.attach_output_handler(
            trainer,
            event_name=ieng.Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda loss: {"loss": loss}
        )
        
        #Save checkpoints
        handler = Checkpoint(
            {'model': model, 'optimizer': optimizer, 'trainer': trainer},
            DiskSaver(f'{saved_models_path}/checkpoints', create_dir=True, require_empty=False),
            n_saved=2,  # Số lượng checkpoint cần lưu
            global_step_transform=lambda e, _: e.state.epoch  # Sử dụng số epoch làm global step
        )
        trainer.add_event_handler(ieng.Events.EPOCH_COMPLETED, handler)
        
        #Load checkpoints
        if isinstance(model_args['pretrained'], str):
            checkpoint = torch.load(model_args['pretrained'])
            Checkpoint.load_objects(to_load={
                'model': model,
                'optimizer': optimizer,
                'trainer': trainer
            }, checkpoint=checkpoint)
            print(f"Load checkpoint: {model_args['pretrained']}")
        
        # Events
        if not skip_train_val:
            @trainer.on(ieng.Events.STARTED)
            def engine_started(engine: ieng.Engine):
                log_validation(engine, False)

        @trainer.on(ieng.Events.EPOCH_STARTED)
        def reset_progress_iterations(engine: ieng.Engine):
            prog_bar_iters.clear()
            prog_bar_iters.n = 0
            prog_bar_iters.last_print_n = 0
            prog_bar_iters.start_t = time.time()
            prog_bar_iters.last_print_t = time.time()
            prog_bar_iters.total = len(engine.state.dataloader)

        @trainer.on(ieng.Events.ITERATION_COMPLETED)
        def log_training(engine: ieng.Engine):
            prog_bar_iters.update(1)

            num_iter = (engine.state.iteration - 1) % len(train_loader) + 1

            early_stop = np.isnan(engine.state.output) or np.isinf(engine.state.output)

            if num_iter % log_interval == 0 or num_iter == len(train_loader) or early_stop:
                tqdm.tqdm.write(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.4f}'.format(
                        engine.state.epoch, num_iter, len(train_loader), engine.state.output
                    )
                )

            if early_stop:
                tqdm.tqdm.write(colored('Early stopping due to invalid loss value.', 'red'))
                trainer.terminate()

        def log_validation(engine: ieng.Engine,
                           train: bool = True):
            if train:
                run_type = 'Train.'
                data_loader = train_loader
                validator = validator_train
            else:
                run_type = 'Eval.'
                data_loader = eval_loader
                validator = validator_eval

            prog_bar_validation = tqdm.tqdm(
                data_loader,
                desc=f'Validation {run_type}',
                file=orig_stdout,
                dynamic_ncols=True,
                leave=False
            )
            validator.run(prog_bar_validation)
            prog_bar_validation.clear()
            prog_bar_validation.close()

            tqdm_info = [
                'Epoch: {}'.format(engine.state.epoch)
            ]
            
            for metric_detail in performance_metrics:
                if train and "val" in metric_detail["use_for"]:
                    metric_label = metric_detail["label"]
                    tqdm_info.append('{}: {:.4f}'.format(metric_label, validator.state.metrics[metric_label]))
                elif not train and "test" in metric_detail["use_for"]:
                    metric_label = metric_detail["label"]
                    tqdm_info.append('{}: {:.4f}'.format(metric_label, validator.state.metrics[metric_label]))
            
            tqdm.tqdm.write('{} results - {}'.format(run_type, '; '.join(tqdm_info)))
        
        if not skip_train_val:
            @trainer.on(ieng.Events.EPOCH_COMPLETED)
            def log_validation_train(engine: ieng.Engine):
                log_validation(engine, True)

        @trainer.on(ieng.Events.EPOCH_COMPLETED)
        def log_validation_eval(engine: ieng.Engine):
            log_validation(engine, False)

            prog_bar_epochs.update(1)

            if scheduler is not None:
                scheduler.step(engine.state.epoch)

        trainer.run(train_loader, max_epochs=epochs)

        del train_loader
        del eval_loader

        prog_bar_iters.clear()
        prog_bar_iters.close()

        prog_bar_epochs.clear()
        prog_bar_epochs.close()

    tqdm.tqdm.write('\n')


def main():
    with _utils.tqdm_stdout() as orig_stdout:
        parser = argparse.ArgumentParser()

        parser.add_argument('-c', '--config', type=str, required=True)
        parser.add_argument('-b', '--batch-train', type=int, required=False)
        parser.add_argument('-B', '--batch-test', type=int, required=False)
        parser.add_argument('-w', '--workers-train', type=int, required=False)
        parser.add_argument('-W', '--workers-test', type=int, required=False)
        parser.add_argument('-e', '--epochs', type=int, required=False)
        parser.add_argument('-L', '--log-interval', type=int, required=False)
        parser.add_argument('-M', '--saved-models-path', type=str, required=False)
        parser.add_argument('-R', '--random-seed', type=int, required=False)
        parser.add_argument('-s', '--suffix', type=str, required=False)
        parser.add_argument('-S', '--skip-train-val', action='store_true', default=False)
        parser.add_argument('-V', '--visdom', action='store_true', default=False)

        args, unknown_args = parser.parse_known_args()

        if args.batch_test is None:
            args.batch_test = args.batch_train

        if args.random_seed is not None:
            args.suffix = '{}r-{}'.format(
                '{}_'.format(args.suffix) if args.suffix is not None else '',
                args.random_seed
            )

            np.random.seed(args.random_seed)
            torch.random.manual_seed(args.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.random_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        configs_found = list(sorted(glob.glob(os.path.expanduser(args.config))))
        prog_bar_exps = tqdm.tqdm(
            configs_found,
            desc='Experiments',
            unit='setup',
            file=orig_stdout,
            dynamic_ncols=True
        )

        for config_path in prog_bar_exps:
            config = json.load(open(config_path))

            if unknown_args:
                tqdm.tqdm.write('\nParsing additional arguments...')

            args_not_found = list()
            for arg in unknown_args:
                if arg.startswith('--'):
                    keys = arg.strip('-').split('.')

                    section = config
                    found = True
                    for key in keys:
                        if key in section:
                            section = section[key]
                        else:
                            found = False
                            break

                    if found:
                        override_parser = argparse.ArgumentParser()

                        section_nargs = None
                        section_type = type(section) if section is not None else str

                        if section_type is bool:
                            if section_type is bool:
                                def infer_bool(x: str) -> bool:
                                    return x.lower() not in ('0', 'false', 'no')

                                section_type = infer_bool

                        if isinstance(section, Iterable) and section_type is not str:
                            section_nargs = '+'
                            section_type = {type(value) for value in section}

                            if len(section_type) == 1:
                                section_type = section_type.pop()
                            else:
                                section_type = str

                        override_parser.add_argument(arg, nargs=section_nargs, type=section_type)
                        overridden_args, _ = override_parser.parse_known_args(unknown_args)
                        overridden_args = vars(overridden_args)

                        overridden_key = arg.strip('-')
                        overriding_value = overridden_args[overridden_key]

                        section = config
                        old_value = None
                        for i, key in enumerate(keys, 1):
                            if i == len(keys):
                                old_value = section[key]
                                section[key] = overriding_value
                            else:
                                section = section[key]

                        tqdm.tqdm.write(
                            colored(f'Overriding "{overridden_key}": {old_value} -> {overriding_value}', 'magenta')
                        )
                    else:
                        args_not_found.append(arg)

            if args_not_found:
                tqdm.tqdm.write(
                    colored(
                        '\nThere are unrecognized arguments to override: {}'.format(
                            ', '.join(args_not_found)
                        ),
                        'red'
                    )
                )

            config = defaultdict(None, config)

            experiment_name = config['Setup']['name']

            batch_train = int(_utils.arg_selector(
                args.batch_train, config['Setup']['batch_train'], BATCH_TRAIN
            ))
            batch_test = int(_utils.arg_selector(
                args.batch_test, config['Setup']['batch_test'], BATCH_TEST
            ))
            workers_train = _utils.arg_selector(
                args.workers_train, config['Setup']['workers_train'], WORKERS_TRAIN
            )
            workers_test = _utils.arg_selector(
                args.workers_test, config['Setup']['workers_test'], WORKERS_TEST
            )
            epochs = _utils.arg_selector(
                args.epochs, config['Setup']['epochs'], EPOCHS
            )
            log_interval = _utils.arg_selector(
                args.log_interval, config['Setup']['log_interval'], LOG_INTERVAL
            )
            saved_models_path = _utils.arg_selector(
                args.saved_models_path, config['Setup']['saved_models_path'], SAVED_MODELS_PATH
            )

            model_class = config['Model']['class']
            model_args = config['Model']['args']

            optimizer_class = config['Optimizer']['class']
            optimizer_args = config['Optimizer']['args']

            if 'Scheduler' in config:
                scheduler_class = config['Scheduler']['class']
                scheduler_args = config['Scheduler']['args']
            else:
                scheduler_class = None
                scheduler_args = None

            dataset_class = config['Dataset']['class']
            dataset_args = config['Dataset']['args']

            transforms = config['Transforms']
            performance_metrics = config['Metrics']

            tqdm.tqdm.write(f'\nStarting experiment "{experiment_name}"\n')

            run(
                experiment_name=experiment_name,
                model_class=model_class,
                model_args=model_args,
                optimizer_class=optimizer_class,
                optimizer_args=optimizer_args,
                dataset_class=dataset_class,
                dataset_args=dataset_args,
                batch_train=batch_train,
                batch_test=batch_test,
                workers_train=workers_train,
                workers_test=workers_test,
                transforms=transforms,
                epochs=epochs,
                log_interval=log_interval,
                saved_models_path=saved_models_path,
                performance_metrics=performance_metrics,
                scheduler_class=scheduler_class,
                scheduler_args=scheduler_args,
                model_suffix=config['Setup']['suffix'],
                setup_suffix=args.suffix,
                orig_stdout=orig_stdout,
                skip_train_val=args.skip_train_val,
            )

        prog_bar_exps.close()

    tqdm.tqdm.write('\n')
