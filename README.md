# Shadow Diffusion for Maintaining Perceptual Quality

This is a modified version of the code base for the following [paper]( https://arxiv.org/pdf/2207.14626.pdf ) to train and execute patch based diffusion model inference for image restoration under shadow conditions.

"Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models"\
<em>Ozan Özdenizci, Robert Legenstein</em>\
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023.\
https://doi.org/10.1109/TPAMI.2023.3238179

If you prefer not to spend time reviewing the code and training the model using the dataset, please follow the instructions provided in the **Configuration**, **Dataset** and **Restoration** guides to modify the paths and execute the code directly.

## Configuration

Please navigate to the directory of the ShadowDiffusion source code and locate the file `configs/ntire1.yml` and `configs/ntire2.yml`. Then, modify the content under `data: data_dir` to match the path to your data folder. For example, if your final test dirctionary is `/root/ntire24/ShadowDiffusion`, you should modify like this:

```yml
data：
	data_dir: "/root/ntire24/ShadowDiffusion"
```

## Datasets

We use [24train_input](https://codalab.lisn.upsaclay.fr/my/datasets/download/16dad948-3dc2-478a-9d8f-96c67736da49), [24train_gt](https://codalab.lisn.upsaclay.fr/my/datasets/download/64b00188-5774-47c7-b7fa-b6f76544d531) and 23train_input, 23train_gt announced by the organizer, [24train_input_raw](https://drive.google.com/drive/folders/1IGSfPDwg2el2dGi6sjuOsCAASHm29xKX?usp=sharing) (24train_preliminary_restoration via NAFNet), [24train_gt_raw](https://codalab.lisn.upsaclay.fr/my/datasets/download/64b00188-5774-47c7-b7fa-b6f76544d531) (in fact is 24train_gt) and [Augmask_Red](https://drive.google.com/drive/folders/1AK_zRiFKS4aOBtHEBv25_V4tf272fWeK?usp=sharing) (generating shadows for specific images of red towel type as a dataset) and [test](https://drive.google.com/drive/folders/1-Q-OwctHHTWZqt9hb_0wVERsIEUQemoY?usp=sharing) (24final_test_preliminary_restoration via NAFNet). 

Arrange the folder structure according to the contents of [train.txt](https://drive.google.com/file/d/1la9o8HU4SehS0AatOMDf1TwTwA15hqoe/view?usp=sharing), [valid.txt](https://drive.google.com/file/d/1nqyHc3DUjaYCydXzw-PthensI_lC85Ej/view?usp=sharing) and [test1.txt](https://drive.google.com/file/d/1pWOSw9O3I2s470OdYomrtfVj07RRejMU/view?usp=sharing) , [test2.txt](https://drive.google.com/file/d/1wXsS7aCjWs3ZzbVn_oW16U-6CA__JMP-/view?usp=sharing) .

```yaml
data:
  - 24:
      - input:
      - gt:
      - input_raw:
      - gt_raw:
      - valid:
  -23:
      - input:
      - gt:
  - Augmask_Red:
      - w.oInputEnsemble_Augmask_Red:
          - images:
  - test:
  - train.txt
  - valid.txt
  - test.txt
```

If you intend solely for **inference**, then you **only** need to download these files: `test`, `train.txt`, `test1.txt` and `test2.txt`.

## Train

To train ShadowDiff<sub>1</sub> and ShadowDiff<sub>2</sub> (generalize to specific scenario), you can use the following command. However, please note that training the model can be time-consuming. It is recommended to utilize pre-trained models in Restoration section whenever possible.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_diffusion.py --config "ntire1.yml" --resume "ShadowDiff1_2000epochs.pth.tar" --sampling_timesteps 25 --seed 61
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_diffusion.py --config "ntire2.yml" --resume "ShadowDiff2_2660epochs.pth.tar" --sampling_timesteps 25 --seed 61
```

## Restoration

We share a pre-trained diffusive **multi-shadow** restoration model [ShadowDiff1](https://drive.google.com/file/d/1cm6MC5wxBBvr-wLsSZXE9cB0ZEAZ_Ka2/view?usp=sharing) and [ShadowDiff2](https://drive.google.com/file/d/1s4sNA9hLQOOxG5mx5JEmooVNRwV8lmEB/view?usp=sharing) with the network configuration in `configs/ntire1.yml` and `configs/ntire2.yml`. Then place ShadowDiff<sub>64</sub> in the root directory of the original code. To evaluate ShadowDiff<sub>64</sub> using the pre-trained model checkpoint with the current version of the repository: 

```bash
CUDA_VISIBLE_DEVICES=0 python eval_diffusion.py --config1 "ntire1.yml" --config2 "ntire2.yml" --resume1 'ShadowDiff1_2000epochs.pth.tar' --resume2 'ShadowDiff2_2660epochs.pth.tar' --test_set 'finaltest' --sampling_timesteps 250 --grid_r 16
```

For slightly improved results and enhanced image quality, consider using a larger value for `sampling_timesteps` and a smaller value for `grid_r`. For instance, the following command yields better image restoration results compared to the one provided earlier. Keep in mind that achieving better visual outcomes may require more computational time, so please be patient during the process.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_diffusion.py --config1 "ntire1.yml" --config2 "ntire2.yml" --resume1 'ShadowDiff1_2000epochs.pth.tar' --resume2 'ShadowDiff2_2660epochs.pth.tar' --test_set 'finaltest' --sampling_timesteps 250 --grid_r 8
```

Finally, the [final result](https://drive.google.com/drive/folders/1n5Ik0P_4JLDkyRpK9oTfBMdK_9X3LSNY?usp=sharing) is obtained by taking the weighted average of the test and finaltest, resulting in a significant improvement over the [final_test_data](https://codalab.lisn.upsaclay.fr/my/datasets/download/cc787344-dada-41b2-9a31-d789f26aa1e4).

```bash
python weight.py
```

## Reference

The citation for the paper from which the WeatherDiffusion code is used is as follows:
```
@article{ozdenizci2023,
  title={Restoring vision in adverse weather conditions with patch-based denoising diffusion models},
  author={Ozan \"{O}zdenizci and Robert Legenstein},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  pages={1-12},
  year={2023},
  doi={10.1109/TPAMI.2023.3238179}
}
```

## Acknowledgments

Parts of this code repository is based on the following works:

* https://github.com/ermongroup/ddim
* https://github.com/bahjat-kawar/ddrm
* https://github.com/JingyunLiang/SwinIR

Please feel free to contact me via email (ferapont@mail.ustc.edu.cn) if you have any inquiries, and I will endeavor to assist you to the best of my abilities.
