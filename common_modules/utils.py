import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

digits = 4


def get_batch(loader_iters):
    batch = []
    for iter in loader_iters:
        data = next(iter)

        data["input_ids"] = data["input_ids"].cuda()
        data["token_type_ids"] = data["token_type_ids"].cuda()
        data["attention_mask"] = data["attention_mask"].cuda()
        data["input_ids_aug"] = data["input_ids_aug"].cuda()
        data["token_type_ids_aug"] = data["token_type_ids_aug"].cuda()
        data["attention_mask_aug"] = data["attention_mask_aug"].cuda()
        data["image"] = data["image"].cuda()
        data["image_aug"] = data["image_aug"].cuda()
        data["image_mask"] = data["image_mask"].cuda()
        data["label"] = data["label"].cuda()

        batch.append(data)
    return batch


def prepare_text(bert_model, text_data, aug=False):
    text = bert_model(
        input_ids=text_data["input_ids"],
        attention_mask=text_data["attention_mask"],
        token_type_ids=text_data["token_type_ids"])
    if aug:
        augmented_text = bert_model(
            input_ids=text_data["input_ids_aug"],
            attention_mask=text_data["attention_mask_aug"],
            token_type_ids=text_data["token_type_ids_aug"])
        return text[1], augmented_text[1]  # (8, 768)
    else:
        return text[1]  # (8, 768)


def prepare_image(model_image, image_data, image_mask):
    b, l, c, h, w = image_data.size()
    image = []
    for idx in range(b):
        _image = []
        for img, i in zip(image_data[idx], image_mask[idx]):
            if i == 1:
                _image.append(img)
        _image = torch.stack(_image, dim=0)
        image += [torch.sum(model_image(_image), dim=0) / sum(image_mask[idx])]
    image = torch.stack(image, dim=0)

    return image  # (8,2048)


def eval_on(
        cfg,
        model,
        val_loaders,
        best_acc,
        logger,
        epoch,
        best_epoch=-1,
        ckpt_prefix='',
        save_best=True
):
    logger.info("start evaluation")
    acc = validate(model, val_loaders, logger, epoch)

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "best_acc": best_acc,
    }

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        if save_best:
            torch.save(
                state, osp.join(cfg["snapshot_dir"], f"{ckpt_prefix}ckpt_best.pth")
            )

    logger.info(
        "\033[31m * Currently, the best val result is: {:.2f}, best epoch is : {}\033[0m".format(
            best_acc * 100, best_epoch
        )
    )

    return best_acc


def validate(
        model,
        data_loaders,
        logger,
        epoch
):
    model.eval()

    data_loader = data_loaders[-1]
    loader_length = len(data_loader)
    loader_iter = [iter(data_loader)]

    pred_list = []
    gt_list = []
    for step in range(loader_length):
        batch = get_batch(loader_iter)[0]

        with torch.no_grad():
            pred = model.forward_test(batch)
            pred_label = torch.argmax(pred, dim=1)

            pred_list += pred_label.cpu().numpy().tolist()
            gt_list += batch["label"].cpu().numpy().tolist()

    report = classification_report(gt_list, pred_list, digits=digits)
    report_dict = classification_report(gt_list, pred_list, output_dict=True, digits=digits)
    if epoch is not None:
        logger.info(f'epoch = {epoch}')
    print(report)
    logger.info(report)
    acc = report_dict['accuracy']
    return acc


def validate_double_model(
        model_text,
        model_image,
        data_loaders,
        logger,
        epoch=None
):
    model_text.eval()
    model_image.eval()

    data_loader = data_loaders[-1]
    loader_length = len(data_loader)
    loader_iter = [iter(data_loader)]

    pred_list_text = []
    pred_list_image = []
    pred_list_fusion = []
    gt_list = []
    for step in range(loader_length):
        batch = get_batch(loader_iter)[0]

        with torch.no_grad():
            pred_text = model_text.forward_test(batch)
            pred_image = model_image.forward_test(batch)
            if type(pred_text) == tuple:
                pred_text, _ = pred_text
                pred_image, _ = pred_image
            pred_text = pred_text['seg_logit']
            pred_image = pred_image['seg_logit']
            pred_text_label = torch.argmax(pred_text, dim=1)
            pred_image_label = torch.argmax(pred_image, dim=1)

            # softmax average (ensembling)
            probs_2d = F.softmax(pred_text, dim=1)
            probs_3d = F.softmax(pred_image, dim=1)

            rate = 1

            pred_label_voxel_ensemble = (probs_2d + rate * probs_3d).argmax(1).cpu().numpy().tolist()
            pred_list_fusion += pred_label_voxel_ensemble

            pred_list_text += pred_text_label.cpu().numpy().tolist()
            pred_list_image += pred_image_label.cpu().numpy().tolist()
            gt_list += batch["label"].cpu().numpy().tolist()
    # text
    report = classification_report(gt_list, pred_list_text, digits=digits)
    report_dict = classification_report(gt_list, pred_list_text, output_dict=True, digits=digits)
    if epoch is not None:
        logger.info(f'epoch = {epoch}')
    print(report)
    logger.info(report)
    acc_text = report_dict['accuracy']

    # image
    report = classification_report(gt_list, pred_list_image, digits=digits)
    report_dict = classification_report(gt_list, pred_list_image, output_dict=True, digits=digits)
    if epoch is not None:
        logger.info(f'epoch = {epoch}')
    print(report)
    logger.info(report)
    acc_image = report_dict['accuracy']

    # fusion
    report = classification_report(gt_list, pred_list_fusion, digits=digits)
    report_dict = classification_report(gt_list, pred_list_fusion, output_dict=True, digits=digits)
    if epoch is not None:
        logger.info(f'epoch = {epoch}')
    print(report)
    logger.info(report)
    acc_fusion = report_dict['accuracy']

    return acc_text, acc_image, acc_fusion


def eval_double_model(
        cfg,
        model_text,
        model_image,
        val_loaders,
        best_acc_text,
        best_acc_image,
        logger,
        best_epoch_text=0,
        best_epoch_image=0,
        epoch=-1,
        save_best=True
):
    logger.info("start evaluation")

    acc_text, acc_image, acc = validate_double_model(model_text, model_image, val_loaders, logger, epoch)

    # text
    state = {
        "epoch": epoch,
        "model_state": model_text.state_dict(),
        "best_acc": acc_text,
    }

    if acc_text > best_acc_text:
        best_acc_text = acc_text
        best_epoch_text = epoch
        if save_best:
            torch.save(
                state, osp.join(cfg["snapshot_dir"], "ckpt_text_best.pth")
            )

    # image
    state = {
        "epoch": epoch,
        "model_state": model_image.state_dict(),
        "best_acc": acc_image,
    }

    if acc_image > best_acc_image:
        best_acc_image = acc_image
        best_epoch_image = epoch
        if save_best:
            torch.save(
                state, osp.join(cfg["snapshot_dir"], "ckpt_image_best.pth")
            )

    logger.info(
        "\033[31m * Currently, the best text val result is: {:.2f}, best epoch is : {}\033[0m".format(
            best_acc_text * 100, best_epoch_text
        )
    )

    logger.info(
        "\033[31m * Currently, the best image val result is: {:.2f}, best epoch is : {}\033[0m".format(
            best_acc_image * 100, best_epoch_image
        )
    )

    return best_acc_text, best_acc_image
