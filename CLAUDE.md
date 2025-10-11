# Mini-Pic

This Rust / Burn repo is designed to enable hyper-efficient, high-quality 64x64 image generation using modern diffusion methods.

Initially, this codebase was copied from a previous model I had worked on. The previous model was somewhat different. It was an RNN / VAE-LSTM for generating time-series. Now we must change it to a rather different model, but using similar patterns.

We need a preprocessor in the `bin` folder for transforming all images in `../diffusiondb/unzipped/` to smaller 64x64 images in `../diffusiondb/unzipped-64/`.

The dataset is diffusiondb, and has been pre-downloaded. We have a set of json files, called parts, inside `../diffusiondb/unzipped-json/` like `part-000001.json` and so on. Each part describes some information associated with an image, and the images are located in `../diffusiondb/unzipped-64/`. It is most likely efficient to load the json first, as it has keys which will indicate which image to load. There are about 1000 images described in each json file.

Example of a diffusiondb json part file:

```
{"ec9b5e2c-028e-48ac-8857-a52814fd2a06.png": {"p": "doom eternal, game concept art, veins and worms, muscular, crustacean exoskeleton, chiroptera head, chiroptera ears, mecha, ferocious, fierce, hyperrealism, fine details, artstation, cgsociety, zbrush, no background ", "se": 3312523387, "c": 7.0, "st": 50, "sa": "k_euler"}, "cd2a819b-faff-410a-af58-b371bd03c587.png": {"p": "a beautiful photorealistic painting of cemetery urbex unfinished building building industrial architecture nature abandoned by thomas cole, nature extraterrestial tron forest darkacademia thermal vision futuristic tokyo, archdaily, wallpaper, highly detailed, trending on artstation. ", "se": 3602562681, "c": 15.0, "st": 50, "sa": "k_lms"}}
```

After we have our preprocessor prepared, we will want to work on the dataloader and figure out the ideal shape of our data. We will want to keep in mind that we are looking to generate images based on user input prompts.

Then comes the bulk of the work, the model.rs file. Here we will implement the foundational and modern diffusion methods.

Later comes optimizing hyperparameters and tweaking training.rs.

Please keep in mind that this original codebase was for Burn 0.15 but now we have upgraded to 0.18 so there will be some minor changes.
