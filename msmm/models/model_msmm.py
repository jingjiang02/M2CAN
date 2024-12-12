import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, balanced_accuracy_score
from transformers import BertModel

from common_modules.base.resnet import resnet50
from common_modules.modules import ImageFeatureExtractor, TextFeatureExtractor, TaskClassifier, MLB
from common_modules.utils import prepare_text, prepare_image
from utils.loss_helper import multi_domain_loss, CriterionMultiModal


class MsmmModel(nn.Module):
    def __init__(self, cfg):
        super(MsmmModel, self).__init__()

        self.bert_model = BertModel.from_pretrained(cfg["dataset"]["bert_model"])

        self.model_image = resnet50(pretrained=True)
        self.model_image.fc = nn.Linear(2048, 2048)
        torch.nn.init.eye_(self.model_image.fc.weight)

        self.text_feature_extractor = TextFeatureExtractor(projection=False)
        self.image_encoder = ImageFeatureExtractor(projection=False)
        self.mlb = MLB()

        self.num_class = int(cfg['net']['num_classes'])
        self.num_domain = len(cfg['dataset']['types'])

        self.source_domains = self.num_domain - 1
        self.target_name = cfg["dataset"]["types"][-1]
        self.origin_json_path = cfg["dataset"]["train"][self.target_name]["data_list"]
        self.pseudo_path = f'{cfg["save_path"]}/pseudo_{self.target_name}_pretrain.json'

        self.classifiers = nn.ModuleList()
        for domain in range(self.source_domains):
            self.classifiers.append(TaskClassifier(out_dim=self.num_class))
        self.text_cls = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 64),
                                      TaskClassifier(out_dim=self.num_class))
        self.image_cls = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 64),
                                       TaskClassifier(out_dim=self.num_class))

        self.task_criterion = nn.CrossEntropyLoss()
        self.multi_modal_loss = CriterionMultiModal()

        self.cross_modal_contrastive_loss_rate = float(cfg["criterion"]["cross_modal_contrastive_loss_rate"])
        self.cross_domain_contrastive_loss_rate = float(cfg["criterion"]["cross_domain_contrastive_loss_rate"])

        # Used to calculate output differences
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.kl_distance = nn.KLDivLoss(reduction='none')

    def forward(self, train_datas):
        # ----------Firstly, we use a pair of pre-trained image and text encoders to project images and text from different domains into a continuous latent feature space----------
        num_datasets = len(train_datas)
        # prepare text and image
        text_batches, augmented_text_batches, image_batches, augmented_image_batches = [], [], [], []
        for data in train_datas:
            text, augmented_text = prepare_text(self.bert_model, data, aug=True)
            text_batches.append(text)
            augmented_text_batches.append(augmented_text)

            image_batches.append(prepare_image(self.model_image, data["image"], data["image_mask"]))
            augmented_image_batches.append(prepare_image(self.model_image, data["image_aug"], data["image_mask"]))

        # extract feature of text
        text_feature_batches = []
        augmented_text_feature_batches = []
        for domain in range(num_datasets):
            text_batch = text_batches[domain]
            augmented_text_batch = augmented_text_batches[domain]
            text_feature_batch = self.text_feature_extractor(text_batch)
            augmented_text_feature_batch = self.text_feature_extractor(
                augmented_text_batch)
            text_feature_batches.append(text_feature_batch)
            augmented_text_feature_batches.append(augmented_text_feature_batch)

        # extract feature of image
        image_feature_batches = []
        augmented_image_feature_batches = []
        for domain in range(num_datasets):
            image_batch = image_batches[domain]
            augmented_image_batch = augmented_image_batches[domain]
            image_feature_batch = self.image_encoder(image_batch)
            augmented_image_feature_batch = self.image_encoder(augmented_image_batch)
            image_feature_batches.append(image_feature_batch)
            augmented_image_feature_batches.append(augmented_image_feature_batch)

        # ----------Secondly, we perform different alignment to learn domain invariant multimodal representations----------
        mm_features = [self.mlb(text_feature_batches[domain], image_feature_batches[domain]) for domain in
                       range(num_datasets)]
        mm_features2 = [self.mlb(text_feature_batches[domain], augmented_image_feature_batches[domain]) for domain in
                        range(num_datasets)]
        mm_features3 = [self.mlb(augmented_text_feature_batches[domain], image_feature_batches[domain]) for domain in
                        range(num_datasets)]
        mm_features4 = [self.mlb(augmented_text_feature_batches[domain], augmented_image_feature_batches[domain]) for
                        domain in range(num_datasets)]

        text_preds = []
        image_preds = []
        for idx in range(num_datasets):
            text_preds.append(self.text_cls(text_feature_batches[idx]))
            image_preds.append(self.image_cls(image_feature_batches[idx]))

        # (1) cross-modal contrastive feature alignment (CMCFA)
        cross_modal_contrastive_loss = self.multi_modal_loss(
            text_preds,
            image_preds,
            text_feature_batches,
            augmented_text_feature_batches,
            image_feature_batches,
            augmented_image_feature_batches,
            num_datasets,
        )

        # (2) cross-domain contrastive feature alignment (CDCFA).
        cross_domain_contrastive_loss = multi_domain_loss(
            text_feature_batches,
            augmented_text_feature_batches,
            image_feature_batches,
            augmented_image_feature_batches,
            num_datasets,
        )

        # task loss
        task_loss = 0.
        preds = []
        rates = [1.0, 1.0, 1.0]
        for idx in range(num_datasets):
            rate = rates[idx]
            preds_local = []
            label = train_datas[idx]["label"]
            for cls_idx in range(self.source_domains):
                predict_label = self.classifiers[cls_idx](mm_features[idx])
                preds_local.append(predict_label)
                predict_label2 = self.classifiers[cls_idx](mm_features2[idx])
                predict_label3 = self.classifiers[cls_idx](mm_features3[idx])
                predict_label4 = self.classifiers[cls_idx](mm_features4[idx])

                task_loss += rate * (
                        self.task_criterion(predict_label, label)
                        + self.task_criterion(predict_label2, label)
                        + self.task_criterion(predict_label3, label)
                        + self.task_criterion(predict_label4, label)
                ) / (4 * num_datasets * self.source_domains)
            pred_fusion = torch.stack(preds_local, dim=0)
            pred_fusion = pred_fusion.mean(dim=0).squeeze()

            # new added fusion loss fuc
            task_loss += rate * self.task_criterion(pred_fusion, label) / num_datasets

            preds.append(pred_fusion)
        # modal header loss
        for idx in range(num_datasets):
            rate = rates[idx]
            label = train_datas[idx]["label"]
            task_loss += rate * (
                    self.task_criterion(text_preds[idx], label)
                    + self.task_criterion(image_preds[idx], label)
            ) / (2 * num_datasets)
        # (1.0 / 192.0)
        return self.cross_modal_contrastive_loss_rate * cross_modal_contrastive_loss, self.cross_domain_contrastive_loss_rate * cross_domain_contrastive_loss, [
            mm_features, preds], task_loss

    def forward_test(self, val_data, multi_output=False):
        text = prepare_text(self.bert_model, val_data)
        image = prepare_image(self.model_image, val_data["image"], val_data["image_mask"])

        # extract feature of text
        text_feature = self.text_feature_extractor(text)

        # extract feature of image
        image_feature = self.image_encoder(image)

        mm_feature = self.mlb(text_feature, image_feature)
        preds = [self.classifiers[domain](mm_feature) for domain in range(self.num_domain - 1)]
        pred_fusion = torch.stack(preds, dim=0)
        pred_fusion = pred_fusion.mean(dim=0).squeeze()
        if multi_output:
            return pred_fusion, preds
        else:
            return pred_fusion

    def forward_test_with_feature(self, val_data):
        text = prepare_text(self.bert_model, val_data)
        image = prepare_image(self.model_image, val_data["image"], val_data["image_mask"])

        # extract feature of text
        text_feature = self.text_feature_extractor(text)

        # extract feature of image
        image_feature = self.image_encoder(image)

        mm_feature = self.mlb(text_feature, image_feature)
        preds = [self.classifiers[domain](mm_feature) for domain in range(self.num_domain - 1)]
        pred_fusion = torch.stack(preds, dim=0)
        pred_fusion = pred_fusion.mean(dim=0).squeeze()

        return pred_fusion, mm_feature

    def validate_pseudo(
            self,
            target_dataloader,
            pseudo_rate
    ):
        digits = 4
        pseudo_threshold = 0.0
        self.eval()

        def convert_to_one_hot(Y, C):
            Y = np.eye(C, dtype=np.int32)[Y.reshape(-1)]
            return Y

        loader_length = len(target_dataloader)
        loader_iter = iter(target_dataloader)

        print_iter = loader_length // 10

        predict_dict = {}
        for idx in range(0, loader_length):
            if idx % print_iter == 0:
                print(f'generating pseudo {idx}/{loader_length}')

            batch = next(loader_iter)
            batch["input_ids"] = batch["input_ids"].cuda()
            batch["token_type_ids"] = batch["token_type_ids"].cuda()
            batch["attention_mask"] = batch["attention_mask"].cuda()
            batch["input_ids_aug"] = batch["input_ids_aug"].cuda()
            batch["token_type_ids_aug"] = batch["token_type_ids_aug"].cuda()
            batch["attention_mask_aug"] = batch["attention_mask_aug"].cuda()
            batch["image"] = batch["image"].cuda()
            batch["image_aug"] = batch["image_aug"].cuda()
            batch["image_mask"] = batch["image_mask"].cuda()
            batch["label"] = batch["label"].cuda()

            with torch.no_grad():
                # pred = self.forward_test(batch)
                pred, preds = self.forward_test(batch, multi_output=True)

                # Perform threshold processing
                pred = torch.softmax(pred, 1)

                pred_max = pred > pseudo_threshold
                pred_max = torch.sum(pred_max, 1) >= 1
                pred_label = torch.argmax(pred, dim=1)
                pred_label_with_threshold = pred_label[pred_max]

                # cls score
                pred_numpy = pred.cpu().numpy()
                one_hot_label = convert_to_one_hot(pred_label.cpu().numpy(), pred.shape[-1])
                one_hot_label = one_hot_label == 1
                pred_logit_with_threshold = pred_numpy[one_hot_label]

                # uncertainty score
                variance = torch.sum(self.kl_distance(self.log_sm(preds[0]), self.sm(preds[1])), dim=1) + torch.sum(
                    self.kl_distance(self.log_sm(preds[1]), self.sm(preds[0])), dim=1)
                exp_variance = torch.exp(-variance)
                pred_uncertainty_with_threshold = exp_variance[pred_max].cpu().numpy()

                # fusion
                pred_score_with_threshold = (pred_logit_with_threshold * pred_uncertainty_with_threshold).tolist()

                pred_list_local = pred_label_with_threshold.cpu().numpy().tolist()

                real_i = 0
                for i, id in enumerate(batch["id"]):
                    if not pred_max[i]:
                        continue
                    if f'{id}' in predict_dict:
                        raise Exception(f'{id} already exists')
                    predict_dict[f'{id}'] = [pred_list_local[real_i], pred_score_with_threshold[real_i]]
                    real_i += 1
        # generate pseudo labels
        with open(self.origin_json_path, mode='r', encoding='utf-8') as f:
            origin_json = json.loads(f.read())
        new_json_with_class = {}
        origin_json_id_dict = {}
        for line in origin_json:
            origin_id = '-'.join(line["image"])
            origin_json_id_dict[origin_id] = int(line["label"])
            try:
                pred_line_label, pred_line_logit = predict_dict[f'{origin_id}']
            except KeyError:
                continue
            line["label"] = pred_line_label
            line["logit"] = pred_line_logit
            # Select according to category
            class_list = new_json_with_class.get(pred_line_label, [])
            class_list.append(line)
            new_json_with_class[pred_line_label] = class_list
        new_json_all = []
        pseudo_num_per_class = int(pseudo_rate * len(origin_json) / self.num_class)
        for _class in new_json_with_class:
            new_json = new_json_with_class[_class]
            new_json = sorted(new_json, key=lambda x: x["logit"], reverse=True)
            new_json = new_json[:pseudo_num_per_class]
            new_json_all += new_json

        random.shuffle(new_json_all)
        with open(self.pseudo_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(new_json_all))

        pred_list = []
        gt_list = []
        for line in new_json_all:
            pred_list.append(line["label"])

            origin_id = '-'.join(line["image"])
            gt_list.append(origin_json_id_dict[origin_id])

        report = classification_report(gt_list, pred_list, digits=digits)
        report_dict = classification_report(gt_list, pred_list, output_dict=True, digits=digits)
        balanced_acc = balanced_accuracy_score(gt_list, pred_list)
        acc = report_dict['accuracy']
        print(f'target train set {self.target_name} pseudo label generated!')
        print(report)
        print(
            f'pseudo balanced_acc = {round(balanced_acc, digits)}, acc = {round(acc, digits)}')
