from __future__ import print_function, division
import torch

from deepkin.models.arguments import finetune_args
# @Workstation-PC
from deepkin.models.kinyagpt import KinyaGPT_SequenceClassifier, KinyaGPT_from_pretrained
from deepkin.models.modules import BaseConfig
from deepkin.utils.misc_functions import time_now, read_lines


def cls_model_inference(args, cls_model, shared_encoder, device, eval_dataset, cls_dict, outfile, regression_target, regression_scale_factor):
    from cls_data_loaders import cls_model_predict
    out = open(outfile, 'w')
    out.write("index	prediction\n")

    cls_model = cls_model.to(device)
    cls_model.eval()
    with torch.no_grad():
        for idx,data_item in enumerate(eval_dataset.itemized_data):
            output_scores, predicted_label, fake_true_label = cls_model_predict(data_item, cls_model, shared_encoder, device)
            if (idx > 0):
                out.write("\n")
            if regression_target:
                val = regression_scale_factor * output_scores.item()
                if val > regression_scale_factor:
                    val = regression_scale_factor
                if val < 0.0:
                    val = 0.0
                out.write(str(idx) + "	" + str(val))
            else:
                out.write(str(idx) + "	" + cls_dict[predicted_label])
    out.close()

def GLUE_eval_main(args, cfg: BaseConfig):
    from cls_data_loaders import ClsDataset

    labels = sorted(args.cls_labels.split(','))
    label_dict = {v:k for k,v in enumerate(labels)}
    cls_dict = {k:v for k,v in enumerate(labels)}
    num_classes = len(label_dict)

    USE_GPU = (args.gpus > 0)

    device = torch.device('cpu')
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')

    print(time_now(), 'Preparing test set ...', flush=True)
    test_lines_input0 = read_lines(args.cls_test_input0)
    test_lines_input1 = read_lines(args.cls_test_input1) if (args.cls_test_input1 is not None) else None
    test_dataset = ClsDataset(test_lines_input0, lines_input1=test_lines_input1,
                                   label_dict=label_dict, label_lines=None,
                                   regression_target=args.regression_target,
                                   regression_scale_factor=args.regression_scale_factor,
                                   max_seq_len=args.main_sequence_encoder_max_seq_len)

    print(time_now(), 'Evaluating devbest model ...', flush=True)
    cls_model = KinyaGPT_SequenceClassifier(args, cfg, num_classes)
    if args.encoder_fine_tune:
        shared_encoder = None
    else:
        shared_encoder = KinyaGPT_from_pretrained(args, cfg, args.pretrained_model_file).encoder.to(device)

    kb_state_dict = torch.load(args.devbest_cls_model_save_file_path, map_location=device)
    cls_model.load_state_dict(kb_state_dict['model_state_dict'])

    cls_model_inference(args, cls_model, shared_encoder, device, test_dataset, cls_dict, args.devbest_cls_output_file, args.regression_target, args.regression_scale_factor)

    print(time_now(), 'Done!', flush=True)

if __name__ == '__main__':
    import os
    args = finetune_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1

    cfg = BaseConfig()
    print('BaseConfig: \n\ttot_num_stems: {}\n'.format(cfg.tot_num_stems),
          '\ttot_num_affixes: {}\n'.format(cfg.tot_num_affixes),
          '\ttot_num_lm_morphs: {}\n'.format(cfg.tot_num_lm_morphs),
          '\ttot_num_pos_tags: {}\n'.format(cfg.tot_num_pos_tags), flush=True)

    GLUE_eval_main(args, cfg)
