# VMSVFND
Code for paper["***T3SVFND:Fake News Detection Adapted to Emergencies on Short Video Platforms***"]

### Environment
please refer to the file requirements.txt.

### Data Processing
The training model can directly use the pre-extracted embeddings: ([VGG19](https://drive.google.com/file/d/13zHvkpGSM5s-ycXsJHLzm4SIA6H-HNL_/view?usp=drive_link)/[hubert](https://drive.google.com/file/d/152eKYVI-bumJrqIM2dZ0O-BAnD6o7Usa/view?usp=drive_link)/[C3D](https://drive.google.com/file/d/1Djn_ey_eb-dfRi9Mlgs26qj4qJ6dNmx0/view?usp=drive_link)/[Comments](https://drive.google.com/file/d/1BIc6_E2FeyPlOfx-BjnF6RzyqH2I2h4F/view?usp=drive_link)).
Please place these features in the specified location, which can be customized in dataloader.py.
Pretrained bert-wwm can be downloaded [here](https://drive.google.com/file/d/1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq/view), and the folder is already prepared in the project.
After placing the data, start training the model:
```python
python main.py
```
### Dataset
The original dataset can be applied for [here](https://github.com/ICTMCG/FakeSV) 

