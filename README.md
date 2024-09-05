# VMSVFND
Code for paper["***T3SVFND:Fake News Detection Adapted to Emergencies on Short Video Platforms***"]

### Environment
please refer to the file requirements.txt.

### Data Processing
The training model can directly use the pre-extracted embeddings: ([VGG19](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/ptvgg19_frames.zip)/[hubert](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/c3d.zip)/[VGGish](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/dict_vid_audioconvfea.pkl)/[Comments](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/dict_vid_audioconvfea.pkl)).
Please place these features in the specified location, which can be customized in dataloader.py.
Pretrained bert-wwm can be downloaded [here](https://drive.google.com/file/d/1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq/view), and the folder is already prepared in the project.
After placing the data, start training the model:
```python
python main.py
```
### Dataset
The original dataset can be applied for [here](https://github.com/ICTMCG/FakeSV) 

