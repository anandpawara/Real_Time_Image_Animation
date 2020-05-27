# Real time Image Animation
The Project is real time application in opencv using first order model

# Steps to setup

## Step 1: Create virtual environment

**Python version** : python v3.7.3 or higher

**create virual environment** : ```pip install virtualenv```

**activate virtual environment** : ```virtualenv env```

## Step 2: Activate virtual environment

**For windows** : ```env/Script/activate```

**For Linux** : ```source env/bin/activate```

## Step 3 : Install required modules

**Install modules** : ``` pip install -r requirements.txt ```

**Install pytorch and torchvision** : ```pip install torch===1.0.0 torchvision===0.2.1 -f https://download.pytorch.org/whl/cu100/torch_stable.html ```

## Step 4 : Download cascade file ,weights and model and save in folder named extract

```gdown --id 1wCzJP1XJNB04vEORZvPjNz6drkXm5AUK```
The file is also availible via direct link on Google's Drive:
https://drive.google.com/uc?id=1wCzJP1XJNB04vEORZvPjNz6drkXm5AUK

**On Linux machine** : ```unzip checkpoints.zip```

If on windows platfrom unzip checkpoints.zip using unzipping software like 7zip.

**Delete zip file** : ```rm checkpoints.zip```

## Step 5 : Run the project

**Run application from live camera** : ```python image_animation.py -i path_to_input_file -c path_to_checkpoint```

**Example** : ```python .\image_animation.py -i .\Inputs\Monalisa.png -c .\checkpoints\vox-cpk.pth.tar```

**Run application from video file** : ```python image_animation.py -i path_to_input_file -c path_to_checkpoint -v path_to_video_file```

**Example** : ```python .\image_animation.py -i .\Inputs\Monalisa.png -c .\checkpoints\vox-cpk.pth.tar -v .\video_input\test1.mp4 ```

![test demo](animate.gif)

### TODO:
Tkinter version

Need work on face alignments

Future plans adding deepfake voice and merging with video

Credits
=======
```
@InProceedings{Siarohin_2019_NeurIPS,
  author={Siarohin, Aliaksandr and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  title={First Order Motion Model for Image Animation},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  month = {December},
  year = {2019},
  url = {https://github.com/AliaksandrSiarohin/first-order-model}
}
```
- Original Project
    * [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model)

    If you like this project give your support to original author of this project by giving github star to author's project

- video explanation to the project <br/>
    * [Video explanation by original author](https://www.youtube.com/watch?v=u-0cQ-grXBQ)
    * [Two min papers](https://www.youtube.com/watch?v=mUfJOQKdtAk)    

- try project on google colab
    * [youtube link](https://www.youtube.com/watch?v=RsOJJd1q6Bg&feature=youtu.be)
    * [link to colab version](https://colab.research.google.com/github/AliaksandrSiarohin/first-order-model/blob/master/demo.ipynb)

    For any valueable feedback feel free to contact me on [linkedin](https://www.linkedin.com/in/anand-pawara-8045/)

