import os
from PSF import PSF
from options import * 
from saver import resume, save_img_single
from tqdm import tqdm

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num / 1024 / 1024, 'Trainable': trainable_num}



def main():
    # parse options    
    parser = TestOptions()
    opts = parser.parse()
    # define model, optimiser and scheduler
    device = torch.device("cuda:{}".format(opts.gpu) if torch.cuda.is_available() else "cpu")
    MPF_model = PSF(opts.class_nb).to(device)
    MPF_model = resume(MPF_model, model_save_path=opts.resume, device=device, is_train=False)
    
    # define dataset    
    # test_dataset = MSRSData(opts, is_train=False)
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset,
    #     batch_size=opts.batch_size,
    #     shuffle=False)
    
    # Train and evaluate multi-task network
    # multi_task_tester(test_loader, MPF_model, device, opts)
    test_path = '/root/PycharmProjects/diffusion_for_mmif/output/M3FD_Fusion_MoEFusion_RGB/'
    # test_path = '/root/dataset/imagefusion/MSRS/test'

    if not os.path.exists(opts.result_dir):
        os.mkdir(opts.result_dir)
        os.mkdir(os.path.join(opts.result_dir, 'rgb'))
        os.mkdir(os.path.join(opts.result_dir, 'gray'))
        os.mkdir(os.path.join(opts.result_dir, 'array'))
    task_tester(test_path, MPF_model, device, opts)

def read_img(path, vis_flage):
    img, _, _ = FusionData.imread(path, vis_flage=vis_flage)
    # import pdb;pdb.set_trace()
    return img

def multi_task_tester(test_loader, multi_task_model, device, opts):
    print(get_parameter_number(multi_task_model))
    multi_task_model.eval()
    is_rgb = False ## 用来标记重建的可见光图像是彩色图像还是灰度图像。
    test_bar= tqdm(test_loader)
    seg_metric = SegmentationMetric(opts.class_nb, device=device)
    lb_ignore = [255]
    ## define save dir
    save_root = os.path.join(opts.result_dir, opts.name)
    Fusion_save_dir = os.path.join(save_root, 'MPF', 'tarin', 'MSRS')
    
    # Fusion_save_dir = '/data/timer/Segmentation/SegFormer/datasets/MSRS/MPF'
    os.makedirs(Fusion_save_dir, exist_ok=True)
    Seg_save_dir = os.path.join(save_root, 'Segmentation')
    os.makedirs(Seg_save_dir, exist_ok=True)
    Re_vis_save_dir = os.path.join(save_root, 'Reconstruction_Vis')
    os.makedirs(Re_vis_save_dir, exist_ok=True)
    Re_ir_save_dir = os.path.join(save_root, 'Reconstruction_IR')
    os.makedirs(Re_ir_save_dir, exist_ok=True)
    with torch.no_grad():  # operations inside don't track history
        for it, (img_ir, img_vi, label, img_names) in enumerate(test_bar):

            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            label = label.to(device)           
            Seg_pred, _, _, fused_img, re_vi, re_ir = multi_task_model(img_vi, img_ir)
            # re_vi = torch.clamp(re_vi, 0, 1)
            # re_ir = torch.clamp(re_ir, 0, 1)
            # fused_img = torch.clamp(fused_img, 0, 1)
            # print(torch.min(fused_img), torch.max(fused_img))
            seg_result = torch.argmax(Seg_pred, dim=1, keepdim=True) ## print(seg_result.shape())
            import pdb;pdb.set_trace()
            seg_metric.addBatch(seg_result, label, lb_ignore)
            # conf_mat.update(seg_result.flatten(), label.flatten())
            # compute mIoU and acc
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            # if not is_rgb:
            #     re_vi = YCbCr2RGB(re_vi, vi_Cb, vi_Cr)
            for i in range(len(img_names)):
                img_name = img_names[i]
                seg_save_name = os.path.join(Seg_save_dir, img_name)
                fusion_save_name = os.path.join(Fusion_save_dir, img_name)
                vi_save_name = os.path.join(Re_vis_save_dir, img_name)
                ir_save_name = os.path.join(Re_ir_save_dir, img_name)
                # seg_visualize(seg_result[i, ::].unsqueeze(0).squeeze(dim=1), seg_save_name)
                # save_img_single(fused_img[i, ::], fusion_save_name)
                # save_img_single(re_vi[i, ::], vi_save_name)
                # save_img_single(re_ir[i, ::], ir_save_name)
                test_bar.set_description('Image: {} '.format(img_name))


def task_tester(test_dir, multi_task_model, device, opts):
    img_ir_path = test_dir
    img_vi_path = test_dir
    # img_ir_path = os.path.join(test_dir, 'ir')
    # img_vi_path = os.path.join(test_dir, 'vi')
    # import pdb;pdb.set_trace()
    img_name_list = os.listdir(img_vi_path)
    print(get_parameter_number(multi_task_model))
    multi_task_model.eval()
    is_rgb = False  ## 用来标记重建的可见光图像是彩色图像还是灰度图像。
    seg_metric = SegmentationMetric(opts.class_nb, device=device)
    lb_ignore = [255]
    ## define save dir
    save_root = os.path.join(opts.result_dir, opts.name)
    Fusion_save_dir = os.path.join(save_root, 'MPF', 'tarin', 'MSRS')

    # Fusion_save_dir = '/data/timer/Segmentation/SegFormer/datasets/MSRS/MPF'
    os.makedirs(Fusion_save_dir, exist_ok=True)
    Seg_save_dir = os.path.join(save_root, 'Segmentation')
    os.makedirs(Seg_save_dir, exist_ok=True)
    Re_vis_save_dir = os.path.join(save_root, 'Reconstruction_Vis')
    os.makedirs(Re_vis_save_dir, exist_ok=True)
    Re_ir_save_dir = os.path.join(save_root, 'Reconstruction_IR')
    os.makedirs(Re_ir_save_dir, exist_ok=True)
    color_list = [
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
          [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128], [128, 128, 128], [255, 64, 128]
    ]
    # import pdb;pdb.set_trace()
    with torch.no_grad():  # operations inside don't track history
        for it, image_name in enumerate(img_name_list):

            img_ir_it = os.path.join(img_ir_path, image_name)
            img_vi_it = os.path.join(img_vi_path, image_name)
            img_ir = read_img(img_ir_it, vis_flage=False)
            img_vi = read_img(img_vi_it, vis_flage=True)

            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            # label = label.to(device)
            # import pdb;pdb.set_trace()
            Seg_pred, _, _, fused_img, re_vi, re_ir = multi_task_model(img_vi, img_ir)
            # re_vi = torch.clamp(re_vi, 0, 1)
            # re_ir = torch.clamp(re_ir, 0, 1)
            # fused_img = torch.clamp(fused_img, 0, 1)
            # print(torch.min(fused_img), torch.max(fused_img))

            seg_result = torch.argmax(Seg_pred, dim=1, keepdim=True)  ## print(seg_result.shape())
            # seg_result = torch.argmax(Seg_pred, dim=1)  ## print(seg_result.shape())
            # seg_metric.addBatch(seg_result, label, lb_ignore)
            # conf_mat.update(seg_result.flatten(), label.flatten())
            # compute mIoU and acc
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)

            seg_result = seg_result.squeeze().cpu().numpy()
            new_seg_result = np.zeros([3, seg_result.shape[-2], seg_result.shape[-1]])
            # import pdb;pdb.set_trace()
            for i in range(opts.class_nb):
                # new_seg_result[:, ...] = np.where(seg_result==i+1, color_list[i], new_seg_result[:, ...])
                new_seg_result[0, seg_result==i] = color_list[i][0]
                new_seg_result[1, seg_result==i] = color_list[i][1]
                new_seg_result[2, seg_result==i] = color_list[i][2]

            # import pdb;pdb.set_trace()
            img = Image.fromarray(new_seg_result.transpose(1,2,0).astype(np.uint8))

            save_name = os.path.join(opts.result_dir, 'rgb',image_name)

            img.save(save_name)

            new_img = Image.fromarray(seg_result.astype(np.uint8))
            save_name = os.path.join(opts.result_dir, 'gray',image_name)
            new_img.save(save_name)
            save_name = os.path.join(opts.result_dir, 'array', image_name.split('.')[0])
            np.save(save_name, Seg_pred.squeeze(0).cpu().numpy())
            # if not is_rgb:
            #     re_vi = YCbCr2RGB(re_vi, vi_Cb, vi_Cr)



if __name__ == '__main__':
    main()
