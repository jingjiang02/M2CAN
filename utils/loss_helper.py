import torch
import torch.nn as nn


def multi_domain_loss(
        text_feature_batches,
        augmented_text_feature_batches,
        image_feature_batches,
        augmented_image_feature_batches,
        num_datasets,
):
    cross_domain_contrastive_loss = 0
    for s in range(num_datasets):
        for t in range(s + 1, num_datasets):
            cross_domain_contrastive_loss += -torch.mm(text_feature_batches[s],
                                                       text_feature_batches[t].t().contiguous()).mean()
            cross_domain_contrastive_loss += -torch.mm(text_feature_batches[s],
                                                       augmented_text_feature_batches[t].t().contiguous()).mean()
            cross_domain_contrastive_loss += -torch.mm(augmented_text_feature_batches[s],
                                                       text_feature_batches[t].t().contiguous()).mean()
            cross_domain_contrastive_loss += -torch.mm(augmented_text_feature_batches[s],
                                                       augmented_text_feature_batches[t].t().contiguous()).mean()
            cross_domain_contrastive_loss += -torch.mm(image_feature_batches[s],
                                                       image_feature_batches[t].t().contiguous()).mean()
            cross_domain_contrastive_loss += -torch.mm(image_feature_batches[s],
                                                       augmented_image_feature_batches[t].t().contiguous()).mean()
            cross_domain_contrastive_loss += -torch.mm(augmented_image_feature_batches[s],
                                                       image_feature_batches[t].t().contiguous()).mean()
            cross_domain_contrastive_loss += -torch.mm(augmented_image_feature_batches[s],
                                                       augmented_image_feature_batches[t].t().contiguous()).mean()
    cross_domain_contrastive_loss = cross_domain_contrastive_loss / 8.0
    return cross_domain_contrastive_loss


class CriterionMultiModal(nn.Module):
    def __init__(self):
        super(CriterionMultiModal, self).__init__()
        self.loss_modal = nn.CrossEntropyLoss()
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.kl_distance = nn.KLDivLoss(reduction='none')

    def forward(
            self,
            text_preds,
            image_preds,
            projected_text_feature_batches,
            projected_augmented_text_feature_batches,
            projected_image_feature_batches,
            projected_augmented_image_feature_batches,
            num_datasets,
    ):
        cross_modal_contrastive_loss = 0
        for domain in range(num_datasets):
            pos_sim = torch.exp(
                torch.sum(projected_text_feature_batches[domain]
                          * projected_image_feature_batches[domain], dim=-1)) + \
                      torch.exp(torch.sum(
                          projected_text_feature_batches[domain]
                          * projected_augmented_image_feature_batches[domain], dim=-1)) + \
                      torch.exp(torch.sum(
                          projected_augmented_text_feature_batches[domain]
                          * projected_image_feature_batches[domain], dim=-1)) + \
                      torch.exp(torch.sum(projected_augmented_text_feature_batches[domain] *
                                          projected_augmented_image_feature_batches[domain], dim=-1))
            sim_matrix = torch.sum(
                torch.exp(torch.mm(projected_text_feature_batches[domain],
                                   projected_image_feature_batches[domain].t().contiguous())) + \
                torch.exp(torch.mm(projected_text_feature_batches[domain],
                                   projected_augmented_image_feature_batches[domain].t().contiguous())) + \
                torch.exp(torch.mm(projected_augmented_text_feature_batches[domain],
                                   projected_image_feature_batches[domain].t().contiguous())) + \
                torch.exp(torch.mm(projected_augmented_text_feature_batches[domain],
                                   projected_augmented_image_feature_batches[domain].t().contiguous()))
                , dim=-1
            )
            loss = (- torch.log(pos_sim / sim_matrix))
            variance = torch.sum(self.kl_distance(self.log_sm(text_preds[domain]), self.sm(image_preds[domain])), dim=1)
            exp_variance = torch.exp(-variance)
            assert loss.shape == exp_variance.shape
            loss = torch.mean(loss * exp_variance) + torch.mean(variance)

            cross_modal_contrastive_loss += loss

        return cross_modal_contrastive_loss
