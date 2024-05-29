## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### Install with Docker

Open one terminal:

```
make build-image
```

```
make run
```

That's it.

If you would like to allow visualization across docker container, open another terminal and type:

```
xhost +
```


### Install without Docker
You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:
```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e GroundingDINO
```


Install diffusers:

```bash
pip install --upgrade diffusers[torch]
```

Install osx:

```bash
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
```

Install RAM & Tag2Text:

```bash
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

More details can be found in [install segment anything](https://github.com/facebookresearch/segment-anything#installation) and [install GroundingDINO](https://github.com/IDEA-Research/GroundingDINO#install) and [install OSX](https://github.com/IDEA-Research/OSX)


## Grounded-SAM Playground
Let's start exploring our Grounding-SAM Playground and we will release more interesting demos in the future, stay tuned!

## :open_book: Step-by-Step Notebook Demo
Here we list some notebook demo provided in this project:
- [grounded_sam.ipynb](grounded_sam.ipynb)
- [grounded_sam_colab_demo.ipynb](grounded_sam_colab_demo.ipynb)
- [grounded_sam_3d_box.ipynb](grounded_sam_3d_box)


### :running_man: GroundingDINO: Detect Everything with Text Prompt

:grapes: [[arXiv Paper](https://arxiv.org/abs/2303.05499)] &nbsp; :rose:[[Try the Colab Demo](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb)] &nbsp; :sunflower: [[Try Huggingface Demo](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo)] &nbsp; :mushroom: [[Automated Dataset Annotation and Evaluation](https://youtu.be/C4NqaRBz_Kw)]

Here's the step-by-step tutorial on running `GroundingDINO` demo:

**Step 1: Download the pretrained weights**

```bash
cd Grounded-Segment-Anything

# download the pretrained groundingdino-swin-tiny model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**Step 2: Running the demo**

```bash
python grounding_dino_demo.py
```

<details>
<summary> <b> Running with Python (same as demo but you can run it anywhere after installing GroundingDINO) </b> </summary>

```python
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./groundingdino_swint_ogc.pth")
IMAGE_PATH = "assets/demo1.jpg"
TEXT_PROMPT = "bear."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)
```

</details>
<br>
