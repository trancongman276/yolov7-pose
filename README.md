# Pose Detection base on Yolov7 Deploy ONNX
The original of this project is [Yolov7](https://github.com/WongKinYiu/yolov7)

<img alt="input.jpg" src=".\image\input.jpg" width="500"/> <br/>
<img alt="result.jpg" src=".\results\result.jpg" width="500"/>

## Install
Download model weight [`yolov7-w6-pose-nms.onnx`](https://drive.google.com/file/d/1Fbl7p6CtdEUyLgVHjUCBU3auX-EV9COg/view?usp=sharing)
and put it in `./weights`.

Install required packages:
```commandline
pip install -r requirements.txt
```
***Note***: This environment will allow you to inference on CPU. In case you want to inference
on GPU, install `onnxruntime-gpu` instead.
## Run
```commandline
python run.py --input_path input.jpg --output_path results
```
## License
Follow license of ONNX, Yolov7.