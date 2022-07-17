# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
from numpy.lib.nanfunctions import _remove_nan_1d

import torch
from torch.nn.parameter import Parameter
# from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from quantization.lsq_layer import QuantAct, QuantConv2d, QuantLinear, QuantMultiHeadAct, QuantMuitiHeadLinear, QuantMuitiHeadLinear_in

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, **kwargs):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        bitops = None
        output = model(images)
        if len(output) == 2:
            output, bitops = output[0], output[1]
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if bitops is not None:
            metric_logger.meters['bitops(G)'].update(bitops.item()/1e9, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_tb(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, bitops_scaler = 0.,  budget = 0., output_dir='test', writer=None, total_epochs=1):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    interval = total_epochs // 3 * 100 # if greater than max_iter, clipped to be max_iter.
    n_iters = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # with torch.cuda.amp.autocast():
        bitops = None
        outputs = model(samples)
        if len(outputs) == 2:
            outputs, bitops = outputs[0], outputs[1]
        # loss = criterion(samples, outputs, targets) + bitops_scaler * (bitops - 21.455 * 1e9) ** 2
        # loss = criterion(samples, outputs, targets) + bitops_scaler * bitops
        if bitops is not None:
            loss = criterion(samples, outputs, targets) + bitops_scaler * (torch.clamp(bitops / 1e9 - budget, min=0)) ** 2
        else:
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        # batch_size = samples.shape[0]

        #tensor board record
        if utils.is_main_process() and writer is not None and n_iters == 0:
            global_iters = epoch
            # global_iters = len(data_loader) * epoch + n_iters
            log_tensorboard(model, writer, global_iters, loss_value)

        n_iters += 1
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)

        # this attribute is added by timm on one optimizer (adahessian)
        # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if utils.is_main_process() and n_iters % 100 == 0:
            log_quantization_parameters(model, output_dir)
    
    if utils.is_main_process():
        log_quantization_parameters(model, output_dir)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def initialize_quantization(data_loader, model, device, output_dir, sample_iters=5):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Initialization:'
    if utils.is_main_process():
        with (output_dir / "scales.txt").open("w") as f:
            f.write("weight scales:\n")
            for name, m in model.named_modules():
                if (isinstance(m, QuantLinear) or isinstance(m, QuantConv2d) or isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in)) and m.alpha is not None:
                    print(f"initialize the weight scale for module {name}")
                    m.initialize_scale(device)
                    f.write(name + ': ' + str(m.alpha.data) + '\n')

            # switch to evaluation mode
            model.eval()
            f.write("activation scales:\n")
            n = 0
            for images, target in metric_logger.log_every(data_loader, 1, header):
                n += 1
                if n > sample_iters:
                    break
                images = images.to(device, non_blocking=True)

                # compute output
                # with torch.cuda.amp.autocast():
                output = model(images)
            for name, m in model.named_modules():
                if (isinstance(m, QuantAct) or isinstance(m, QuantMultiHeadAct)) and m.alpha is not None:
                    print(f"initialize the activation scale for module {name}")
                    m.initialize_scale_offset(device)
                    f.write(name + ': ' + str(m.alpha.data) + '\n')
                    if m.offset:
                        f.write("offset" + ': ' + str(m.beta.data) + '\n')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return

def initialize_muitihead_quantization(model, device):
    for name, m in model.named_modules():
        if (isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in) or isinstance(m, QuantMultiHeadAct)) and m.alpha is not None:
            m.nbits = Parameter(torch.ones(m.num_head).to(device) * m.nbits).to(device)
            print(f"Initialize bit-width for {name}, bit:{m.nbits.data}")

@torch.no_grad()
def update_bn(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Update BN:'

    # switch to evaluation mode
    model.train()

    for images, _ in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)

        # compute output
        output = model(images)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def log_quantization_parameters(model, output_dir):
    with (output_dir / "q_param.txt").open("w") as f:
        f.write("weight scales:\n")
        for name, m in model.named_modules():
            if isinstance(m, QuantLinear) or isinstance(m, QuantConv2d) or isinstance(m, QuantMultiHeadAct) or isinstance(m, QuantAct) or isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in):
                if m.alpha is not None:
                    f.write(name + '\n')
                    f.write('--bitwidth: ' + str(m.nbits.data) + '\n')
                    if m.nbits.grad is not None:
                        f.write('--grad: ' + str(m.nbits.grad.data) + '\n')
                    n = m.nbits.round().to(torch.long)
                    f.write('--scales: ' + str(m.alpha.data) + '\n')
                    f.write('--scale_n: ' + str(m.alpha[n-2]) + '\n')
                    if m.alpha.grad is not None:
                        f.write('--grad: ' + str(m.alpha.grad.data) + '\n')
                if isinstance(m, QuantAct) and m.offset:
                    f.write("--offsets: " + str(m.beta.data) + '\n')
                    f.write("--offset_n: " + str(m.beta[n-2]) + '\n')

@torch.no_grad()
def log_tensorboard(model, writer, global_iters, loss_value):
    

    
    # writer.add_scalar('acc1', acc1, global_iters)
    # writer.add_scalar('acc5', acc5, global_iters)
    writer.add_scalar('loss', loss_value, global_iters)
    for name, m in model.named_modules():
        # QuantLinear and QuantConv2d
        if (isinstance(m, QuantLinear) or isinstance(m, QuantConv2d)) and m.alpha is not None:
            nbits = m.nbits.item()
            n = round(nbits)
            writer.add_scalar(name+'/bitwidth_float', nbits, global_iters)
            writer.add_scalar(name+'/bitwidth', n, global_iters)
            writer.add_scalar(name+'/scale', m.alpha[n-2].item(), global_iters)
            writer.add_scalar(name+'/weight', m.weight.norm().item(), global_iters)
            if m.alpha.grad is not None:
                if m.nbits.grad is not None:
                    writer.add_scalar(name+'/bitwidth_float_grad', m.nbits.grad.item(), global_iters)
                writer.add_scalar(name+'/scale_grad', m.alpha.grad[n-2].item(), global_iters)
                r1 =  m.alpha.grad[n-2].abs().item() / m.alpha[n-2].item()
                writer.add_scalar(name+'/r_scale', r1, global_iters)
                writer.add_scalar(name+'/weight_grad', m.weight.grad.norm().item(), global_iters)
                r2 = m.weight.grad.norm().item() / m.weight.norm().item()
                writer.add_scalar(name+'/r_weight', r2, global_iters)
                # writer.add_scalar(name+'/R', r1 / r2, global_iters)
            writer.add_scalar(name+'/weight_numel', m.weight.numel(), global_iters)
            writer.add_scalar(name+'/weight_mean', m.weight.mean(), global_iters)
        
        # QuantAct
        elif isinstance(m, QuantAct) and m.alpha is not None:
            nbits = m.nbits.item()
            n = round(nbits)
            writer.add_scalar(name+'/bitwidth_float', nbits, global_iters)
            writer.add_scalar(name+'/bitwidth', n, global_iters)
            writer.add_scalar(name+'/scale', m.alpha[n-2].item(), global_iters)
            if m.alpha.grad is not None:
                if m.nbits.grad is not None:
                    writer.add_scalar(name+'/bitwidth_float_grad', m.nbits.grad.item(), global_iters)
                writer.add_scalar(name+'/scale_grad', m.alpha.grad[n-2].item(), global_iters)
                r1 =  m.alpha.grad[n-2].abs().item() / m.alpha[n-2].item()
                writer.add_scalar(name+'/r_scale', r1, global_iters)

            if hasattr(m, 'beta') and m.offset:
                writer.add_scalar(name+'/offset', m.beta[n-2].item(), global_iters)
                if m.beta.grad is not None:
                    writer.add_scalar(name+'/offset_grad', m.beta.grad[n-2].item(), global_iters)
                    r3 =  m.beta.grad[n-2].abs().item() / m.beta[n-2].abs().item()
                    writer.add_scalar(name+'/r_offset', r3, global_iters)

        elif isinstance(m, QuantMultiHeadAct) and m.alpha is not None:
            num_head = m.num_head
            bits_float = {}
            bits_int = {}
            scales = {}
            scales_grad = {}
            for i in range(num_head):
                bit_float = m.nbits[i]
                bit_int = bit_float.round().to(torch.long)
                bits_float['h'+str(i)] = bit_float
                bits_int['h'+str(i)] = bit_int
                scales['h'+str(i)] = m.alpha[bit_int-2]
                if m.alpha.grad is not None:
                    scales_grad['h'+str(i)] = m.alpha.grad[bit_int-2]
            writer.add_scalars(name+'/bitwidth_float', bits_float, global_iters)
            writer.add_scalars(name+'/bitwidth', bits_int, global_iters)
            writer.add_scalars(name+'/scale', scales, global_iters)
            if m.alpha.grad is not None:
                writer.add_scalars(name+'/scale_grad', scales_grad, global_iters)

@torch.no_grad()
def head_analysis(model, head_index):
    for name, m in model.named_modules():
        if 'blocks.0.attn' in name and (isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in) or isinstance(m, QuantMultiHeadAct)) and m.alpha is not None:
            m.nbits.data.fill_(8.)
            m.nbits[head_index].data.fill_(2.)
            print(f"turning bit-width for {name} to 2-bit...\nresulting bit:{m.nbits.data}")