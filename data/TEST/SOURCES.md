# TEST dataset — image sources

`TEST` is the smoke-test dataset: two scenes, 168 patches each, shipped with the
repository so the pipeline can be run end to end without downloading anything.

Its images are not from the datasets analysed in the paper, but from the public domain.
- `images/image01.png` — <https://commons.wikimedia.org/wiki/Category:Cats#/media/File:Female-cat-named-Liebe.jpg>
- `images/image02.png` — <https://upload.wikimedia.org/wikipedia/commons/1/16/Ilya_Repin_Unexpected_visitors.jpg>

Both were center-cropped to the 688:524 aspect ratio and resampled to 688×524 for ease.