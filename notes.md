# Image Classification

## Regex

in python a raw string is a string prefixed with an r like `r"\tTab"` if this was printed, Python would ignore the tab (\t) and return "\tTab"

MetaCharacters of Regex: . ^ $ * + ? { } [ ] \ | ( )

when searching via regex you must escape meta characters with a `\`  For example, if I wanted to search for literal periods I would have to use `"\."` in my find argument. 

Basically Regex is highly general string search tool.  I will learn this in full as I go.  

## Loading Data

From here on we will be using the Datablock and dataloaders functionality from fastai libraries.  Just keep in mind that you probably have to write your own dataloader so don't get to attached.




