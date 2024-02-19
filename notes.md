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


## Resnet

We are using resnet34 this case.  I tried resnet18 and it produced okay results so it got me wondering how well my model would do with this data, going to try it out..

## Cross-Entropy Loss Revisited

fastai automatically assigns a loss function, in this case it uses CEL since we are doing multi-category loss

## Viewing activations and labels

Using .one_batch will yield a batch of data from your dataset (dls) in fastai. 

Take this batch and pass it to the model to yield the activations.  (.get_preds)

The predictions are a list of probabilities that add up to 1

## Softmax revisited

Softmax is similar to the Sigmoid curve in that it ensures the output values fall between 0 and 1

In practice, it does so on a *per category* basis.  Meaning, we will need activations for each category in a group.


Say we had activations for predictions of 3's and 7's like:

```
actv = torch.randn((6,2))*2

print(actv)

out: tensor([[ 4.4414, -0.8224],
        [-1.1615,  0.0147],
        [ 0.9840, -1.1415],
        [-1.1832, -0.8416],
        [-2.4945, -0.3733],
        [-2.2916,  1.6909]])



```

simply putting the activations through the sigmoid would not produce what we need.  We need each row to sum to one once passed through the activation function.

we can do this by taking the sigmoid of (column 0 - column 1):

```
3_preds = (actv[:,0] - actv[:,1]).sigmoid()
print(3_preds)

out: tensor([0.9949, 0.2357, 0.8934, 0.4154, 0.1071, 0.0183])
```

The predictions that are returned are those of column 0 (3's).  The predictions for column 1 (7's) are just (1 - col 0 activations)

```
7_preds = 1- 3_preds
print(7_preds)

out: tensor([0.0051, 0.7643, 0.1066, 0.5846, 0.8929, 0.9817])
```

Matching up the outputs we see that the probabilities sum to 1:

```
is three:  tensor([0.9949, 0.2357, 0.8934, 0.4154, 0.1071, 0.0183]) 
is seven:  tensor([0.0051, 0.7643, 0.1066, 0.5846, 0.8929, 0.9817])
```

to extend this, we need a function that does all of this regardless of the number of categories.  This is where Softmax comes in. it looks something like this:

```
def softmax(x):
  return exp(x)/exp(x).sum(dim=1, keepdim=True)
```

Looking back at my MNIST model I implemented softmax in this way:

```
def softmax(preds):
    preds = preds-torch.max(preds)
    return torch.exp(preds)/torch.sum(torch.exp(preds), dim=1).unsqueeze(1)
```
Breaking the function apart we see that first I center the predictions around the maximum value so as to not have large outliers. 

Then we get into the actual calculation.  It takes a tensor of predictions, the length of which is equal to the number of categories.  The output is a tensor of the same length where each value is the exponentiated value divided by the sum of all the exponentiated values in the tensor.  Thus each value represents the proportion of the total prediction space (100% or 1) it takes up.

Of course there are other functions that could achieve the same outcome, however, softmax is the only one who's curve closely matches that of the sigmoid. 

The book says something interesting about the function:  "The exponential also has a nice property: if one of the numbers in our activations is slightly bigger than the others the exponential will amplify this"

This agrees with my using the `preds-torch.max(preds)` logic.  We want *slight* differences not large ones.  The exponential is a powerful function.

There is an issue with this method.  If we were to use this model for inference it would always make a prediction, even if the input provided does not fall under any of the specified categories.  In the inference case we may want to use a collection of binary classifiers using sigmoid so that if the model does not recognize the image it won't even make a prediction.

## Log Likelihood Revisited

The book first goes over selecting the prediction value that corresponds to its target label.  I used a one hot method that I devised myself in my model but now seems like it was a lot of work for something that was rather simple. 

My Log Likelihood portion of CEL:

```
one_hot = torch.zeros(trgt.shape[0], soft.shape[1])
        for i in range(one_hot.size(0)):
            index = trgt[i, 0].item()
            one_hot[i, int(index)] = 1
        loss = -torch.sum(torch.log(soft)*one_hot)
```

Where the book just takes the predictions and indexes through the length of the prediction batch as well as indexes each row with the target tensor like:

```
targ = tensor([1,0,0,1,1,0])

sm_acts = tensor([[0.6, 0.2],
                [0.4, 0.9],
                [0.3, 0.8],
                [0.5, 0.4],
                [0.2, 0.9],
                [0.5, 0.7]])
index = range(len(sm_acts))
```
With my full cross entropy loss function being:

```
def cross_entropy_loss(preds, trgt):
        soft = softmax(preds)
        one_hot = torch.zeros(trgt.shape[0], soft.shape[1])
        for i in range(one_hot.size(0)):
            index = trgt[i, 0].item()
            one_hot[i, int(index)] = 1
        loss = -torch.sum(torch.log(soft)*one_hot)
        return loss
```

I think I can change it to:

```
def cross_entropy_loss(preds, trgt):
        log_soft = torch.log(softmax(preds))
        one_hot = log_soft[range(len(preds)),trgt]
        loss = -torch.sum(one_hot)
        return loss

```

I am going to try and implement this before moving on.  Visit my MNIST project to see if it was successful.  Figured out I needed to flatten my target tensor.

implementation was successful!  I noticed a faster convergence to the ~90% regime using this method, not sure why.  Model still floats around 93-94% accuracy which is much worse than the world class result of 99.98%.  My current goal is to break the 95% regime consistently.  I think the best way to go about this is to add more layers to the model.  I've tried this once before with no success, I couldn't find the right hyperparameters.  

I found it interesting that my current model setup is not compatible with any of the torch CEL functions/classes.  When I tired to use nn.CrossEntropyLoss()/F.cross_entropy the loss would start at some extremely low value ~2 and the model accuracy would barely make it above 14% in my preliminary tests.  This has me thinking about my implementation and how it differs from torch.  Looking at the docs in torch, the main differences are having to do with summing vs. taking the mean.  nn.CrossEntropyLoss defaults to taking the mean of the loss where as my model just uses the sum to train.  Setting reduction to "sum" fixed the issue with implementing nn.CrossEntropyLoss in my model. I am still unsure as to why you would use the mean in your model since it would often be very slow.  

As a note Using nn.CrossEntropyLoss had the same accuracy as my own implementation so i'll stick with mine for now.

Moving on with Chapter 5

## Model Interpretation


Made a classification matrix to see where the model is screwing up.  The model is getting confused in areas that even I would probably not be able get right most of the time.  

## Improving our Model

Instead of declaring our learning rate, it would be nice to dynamically discover the learning rate in some automated way.  Luckily this has already been done.  I do want to implement this myself.

Implemented a simple learning rate finder and using the rate I extracted from it the model actually performed worse!  I was originally using a learning rate of 0.003 and was getting ~94% accuracy.  Now with the learning rate of 0.0002048 I'm getting an accuracy of ~93.5% but over a much longer training time (~5x the number of epochs). 

I think I need to write a tester that actually shows the model making predictions on numbers from the test set.

Re-reading the the section they give the advice to choose between 10x less than the lowest loss achieved or where the loss was clearly decreasing.  I went with the second option but really pushed "clearly decreasing" and chose the rate just before the bottom out point where LR = 0.0016384.  This resulted in an accuracy of 94.28% after training finished.  Definitely better, but I tried the bottom out point of LR = 0.0032768 and achieved slightly worse results.  

Overall, inspecting the learning rate is probably a good idea and will give good direction in terms of how to set your hyperparameters. 

## Finishing out the Notebook

The Fastai library has the ability to find the steepest point on the lr finder curve, might be something I want to try out for myself.

Finding the LR for fine tuning for sure affects performance of the model.

### Unfreezing and transfer learning

Models consist of many layers. Some of the layers contain information we would actually like to keep intact when we do fine tuning and some of it not.  Usually, the final layer is the least useful to us since it was to trained to output whatever classification the model trainers wanted, not us. 

When doing transfer learning, we throw this last layer away and replace it with our own classification layer.  In the case of the pet breeds, it would be a linear layer with 37 outputs. 

remember that this layer will have randomly trained weights, but the rest of the model has useful image classification information. 

So, to achieve fine tuning, we must *freeze*, the layers 0 -> n-1, train layer n with the preceding layers frozen, then unfreeze the model, training all layers together.  

Using the underlying Fastai functionality we can do this manually-ish. 

```
learn = vision_learner(dls, resnet34, metrics= error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(6, lr_max=4e-6)
```
If you then set `lr = learn.lr_find()` you can call `lr.valley` to find the the suggested learning rate.  I usually go about 10x less, it seems to work best for me


### Discriminative Learning Rates

We can further improve the performance of the model through discriminative learning rates.  during transfer learning, deeper layers benefit from a lower learning rate and vice versa for later layers. 

fastai has a slice option that allows for discriminative learning rates. Slice is a range `(low_lr, high_lr)` where the deepest layer gets the low_lr and the last layer will have high_lr.  Every layer in between will lr's multiplicatively equidistant to each other.  



























