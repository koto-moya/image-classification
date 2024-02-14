# Image Classification

## Regex

in python a raw string is a string prefixed with an r like `r"\tTab"` if this was printed, Python would ignore the tab (\t) and return "\tTab"

MetaCharacters of Regex: . ^ $ * + ? { } [ ] \ | ( )

when searching via regex you must escape meta characters with a `\`  For example, if I wanted to search for literal periods I would have to use `"\."` in my find argument. 

Basically Regex is highly general string search tool.  I will learn this in full as I go.  

## Loading Data

From here on we will be using the Datablock and dataloaders functionality from fastai libraries.  Just keep in mind that you probably have to write your own dataloader so don't get to attached.

```
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items = get_image_files,
                 splitter = RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms = Resize(460),
                 batch_tfms = aug_transforms(size = 224, min_scale=0.75))
```
We'll go line by line to get a better grasp on what this data loader is actually doing

- Blocks: Created an image block and a category block.  Looks like it creates a tuple of the raw data and its labels

- get_items: chooses the function to use for gathering the data

- splitter: splits the block into batches

- get_y: Gather the labels

- item_tfms: applies a transformation uniformly to all images. Applies the resize crop described below. 

- batch_tfms: Applies transformations to each batch. Performs all transformations at once so interpolation is only done one time. 

## Presizing

Most deep learning libraries apply image transformations one by one, resizing/augmenting -> interpolating and repeating for all desired transformations.  THis creates unwanted artifacts in the data, null zones, and overall worse quality data.  Fastai took the approach of applying all image transformations (that occur on the GPU) in one step so that interpolation would only be a applied once. This significantly improves the quality of the data used to train the model.  They achieve this performance by implementing a resizing technique that crops the image to larger dimensions.  It resizes the dimensions based on the height or width of the original image, whichever is smaller. This gives the image spare margin to account for the data augmentation in the next step.

This can be done because most image transformations are linear! any combination of linear functions can be reduced to one linear function that describes them all.


## Checking your Datablock (Debugging)

Always check your data before training. 

Look at the images, do the labels correspond to the right images? use google image search to confirm.  You are not a domain expert; in many areas you will be making models, debugging the dat will help you learn about the field.

Running .summary(<path>) on the dataloader shows the process of loading the data step by step.  Good to do before loading the data into the dataloader to check for sizing issues, dtype errors, etc.

Once you think the data looks okay, you should start training immediately.  Do not put off training the model too long.  You want to know how a baseline model performs to find things like: does my data even train the model?  Does the model fulfill my use case parameters with little overhead in terms of model dev?

You want to know these things as soon as possible to rule out any dead ends and wasted work.  

## Pytorch Issue Detour


Keep getting this error:  "RuntimeError: Adaptive pool MPS: input sizes must be divisible by output sizes."

Looks like I'll have to use Google Colab after all.  Starting to wonder if I should return this computer...


We march forward!











