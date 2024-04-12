import os.path as osp
import glob
import cv2
import torch
import RRDBNet_arch as arch

print('Testing ESRGAN')
model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')

test_img_folder = 'ESRGAN/LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

for path in glob.glob(test_img_folder):
    base, ext = osp.splitext(osp.basename(path))
    print('Processing:', base)

    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = img.permute(2, 0, 1)
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = output.permute(1, 2, 0) 
    output = (output * 255.0).round()

    output_path = f'ESRGAN/results/{base}_rlt{ext}'
    cv2.imwrite(output_path, output)

print('Testing complete.')
