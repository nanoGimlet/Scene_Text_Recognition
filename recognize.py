import os

import torch
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(input_model, input_opt, input_converter):
    """ model configuration"""
    if 'CTC' in input_opt.Prediction:
        input_converter = CTCLabelConverter(input_opt.character)
    else:
        input_converter = AttnLabelConverter(input_opt.character)
    input_opt.num_class = len(input_converter.character)

    if input_opt.rgb:
        input_opt.input_channel = 3
    input_model = Model(input_opt)

    print('model input parameters', input_opt.imgH, input_opt.imgW, input_opt.num_fiducial, input_opt.input_channel, input_opt.output_channel,
          input_opt.hidden_size, input_opt.num_class, input_opt.batch_max_length, input_opt.Transformation, input_opt.FeatureExtraction,
          input_opt.SequenceModeling, input_opt.Prediction)
    input_model = torch.nn.DataParallel(input_model).to(device)

    # load model
    print('loading pretrained model from %s' % input_opt.saved_model)
    input_model.load_state_dict(torch.load(
        input_opt.saved_model, map_location=device))

    return input_model, input_opt, input_converter


def recognize(input_model, input_opt, input_converter):
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(
        imgH=input_opt.imgH, imgW=input_opt.imgW, keep_ratio_with_pad=input_opt.PAD)
    demo_data = RawDataset(root=input_opt.image_folder,
                           opt=input_opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=input_opt.batch_size,
        shuffle=False,
        num_workers=int(input_opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    input_model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor(
                [input_opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(
                batch_size, input_opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in input_opt.Prediction:
                preds = input_model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = input_converter.decode(preds_index, preds_size)

            else:
                preds = input_model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = input_converter.decode(
                    preds_index, length_for_pred)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in input_opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    # prune after "end of sentence" token ([s])
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(
                    f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()

    os.remove("./created_data/target_image/sample_trim.png")
