import logging
import argparse
import torch
from lib.config import Config
from lib.experiment import Experiment
import numpy as np
import cv2
def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    parser.add_argument("mode", choices=["train", "test"], help="Train or test?")
    parser.add_argument("--exp_name", help="Experiment name", required=True)
    parser.add_argument("--cfg", help="Config file")
    parser.add_argument("--videoFile", help="Config file")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--epoch", type=int, help="Epoch to test the model on")
    parser.add_argument("--cpu", action="store_true", help="(Unsupported) Use CPU instead of GPU")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to pickle file")
    parser.add_argument("--view", choices=["all", "mistakes"], help="Show predictions")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")
    args = parser.parse_args()
    if args.cfg is None and args.mode == "train":
        raise Exception("If you are training, you have to set a config file using --cfg /path/to/your/config.yaml")
    if args.resume and args.mode == "test":
        raise Exception("args.resume is set on `test` mode: can't resume testing")
    if args.epoch is not None and args.mode == 'train':
        raise Exception("The `epoch` parameter should not be set when training")
    if args.view is not None and args.mode != "test":
        raise Exception('Visualization is only available during evaluation')
    if args.cpu:
        raise Exception("CPU training/testing is not supported: the NMS procedure is only implemented for CUDA")
    return args

def main():
    args = parse_args()
    exp = Experiment(args.exp_name, args, mode=args.mode)
    if args.cfg is None:
        cfg_path = exp.cfg_path
    else:
        cfg_path = args.cfg
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
    model = cfg.get_model()
    model_path = exp.get_checkpoint_path(args.epoch or exp.get_last_checkpoint_epoch())
    logging.getLogger(__name__).info('Loading model %s', model_path)
    model.load_state_dict(exp.get_epoch_model(args.epoch or exp.get_last_checkpoint_epoch()))
    model = model.to(device)
    model.eval()
    test_parameters = cfg.get_test_parameters()
    predictions = []
    exp.eval_start_callback(cfg)
    videoFile = args.videoFile
    videoWriter = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M','P','4','V'), 15, (640, 360))
    cap = cv2.VideoCapture(videoFile)
    frameNum = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame,(640,360))
        images = frame/ 255.
        frameNum = frameNum + 1
        if frameNum % 2 == 0:  # 调整帧数
            images = torch.from_numpy(images).cuda().float()
            images = torch.unsqueeze(images.permute(2, 0, 1),0)
            output = model(images, **test_parameters)
            prediction = model.decode(output, as_lanes=True)
            predictions.extend(prediction)
            img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = cv2.resize(img, (640, 360))
            img_h, _, _ = img.shape
            pad = 0
            if pad > 0:
                img_pad = np.zeros((360 + 2 * pad, 640 + 2 * pad, 3), dtype=np.uint8)
                img_pad[pad:-pad, pad:-pad, :] = img
                img = img_pad
            for i, l in enumerate(prediction[0]):
                points = l.points  #<class 'lib.lane.Lane'>
                points[:, 0] *= img.shape[1]
                points[:, 1] *= img.shape[0]
                points = points.round().astype(int)
                points += pad
                for curr_p, next_p in zip(points[:-1], points[1:]):
                    frame = cv2.line(frame,
                                   tuple(curr_p),
                                   tuple(next_p),
                                   color=(255, 0, 255),
                                   thickness=3)
            cv2.namedWindow("resized", 0)  # 0可以改变窗口大小了
            cv2.imshow("resized", frame)  # 显示视频
            videoWriter.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()