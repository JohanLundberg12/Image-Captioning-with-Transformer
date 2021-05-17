# Image-Captioning-with-Transformer

See the example jupyter for an example test.

To train the model in English run:
```bash
main.py config_files/english_pretrained.txt
```


To test the metrics of the English model run:
```bash
test.py config_files/english_pretrained.text
```


To test a model on a single image run 
```bash
test_image.py {config_file} {image_path}
```


To output predictions for a model on multiple images run
```bash
test_images.py {config_file} {images_dir}
