# Real time Image Animation
The Project is real time application in opencv using first order model

# Steps to setup

## Step 1: Create virtual environment

**create virual environment** : ```pip install virtualenv```

**activate virtual environment** : ```virtualenv env```

## Step 2: Activate virtual environment

**For windows** : ```env/Script/activate```

**For Linux** : ```source env/bin/activate```

## Step 3 : Install required modules

**Install modules** : ``` pip install -r requirements.txts ```

**Install pytorch and torchvision** : ```pip install torch===1.0.0 torchvision===0.2.1 -f https://download.pytorch.org/whl/cu100/torch_stable.html ```

## Step 4 : Download cascade file ,weights and model and save in folder named extract

```gdown --id 1wCzJP1XJNB04vEORZvPjNz6drkXm5AUK```

**On Linux machine** : ```unzip checkpoints.zip```

If on windows platfrom unzip checkpoints.zip using unzipping software like 7zip.

**Delete zip file** : ```rm checkpoints.zip```

## Step 5 : Run the project

**Run application from live camera** : ```python image_animation.py -i path_to_input_file -c path_to_checkpoint```

**Example** : ```python .\image_animation.py -i .\Inputs\Monalisa.png -c .\checkpoints\vox-cpk.pth.tar```

**Run application from video file** : ```python image_animation.py -i path_to_input_file -c path_to_checkpoint -v path_to_video_file```

**Example** : ```python .\image_animation.py -i .\Inputs\Monalisa.png -c .\checkpoints\vox-cpk.pth.tar -v .\video_input\test1.mp4 ```

![test demo](animate.gif)
