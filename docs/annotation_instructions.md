# Annotation instructions

These are the instructions that every human annotator received before labelling the retrieved image–caption pairs of
the SVO-Probes lists. The link to the annotation tool and the screenshots of its interface are omitted.

## Who labelled what

- The image-query (image-to-text) lists of CLIP, BLIP-2, FLAVA and SigLIP2 (4,000 pairs) were labelled by three
  annotators: an expert, who is one of the authors, and two volunteers. All three have tertiary education and are
  non-native speakers of English. The final label is the majority vote (Fleiss' kappa 0.64).
- All other lists (the caption-query lists of these four models and both directions of Qwen3-VL-Embed) were labelled
  by the expert alone.
- The expert also assigned the error type (subject, verb or object) of every pair labelled incorrect.
- Each model's top-10 list was labelled separately, without showing which model had produced it. A pair that several
  models retrieved therefore has several labels; where they conflict, the analysis keeps the first one in the fixed
  model order CLIP, BLIP-2, FLAVA, SigLIP2, Qwen3-VL-Embed.
- A pair is correct when the caption describes the image fully; an incorrect pair carries the role that is wrong.

## Instructions given to the annotators

Your task is to annotate the relation between the image and captions. Could the caption conceivably describe the image?
Does the caption match the image? More than one caption can be correct for a given image. The annotation schema has
three options (Yes - No - ?), and "?" with a meaning that you can not tell. Try to use "?" only in cases when you are
really not sure about the relationship between image and caption.

- **Yes (Relevant)** - The caption could match the image fully.
- **No (Irrelevant)** - The caption contains some part that does not match the image. The subject, verb, object or the
  number is incorrect (if the image shows man and the caption mentions "woman", the pair is incorrect; if the image
  shows a man on a red carpet and the caption mentions "actor", assume general knowledge that the man is in fact an
  actor).
- **? (Can not tell)** - You can not tell. Please choose this option only when you are really in the middle of yes and
  no. Always try to choose "yes" or "no".

The expected time for evaluating one pair of an image and caption is approximately 5 seconds, while each document
contains multiple pairs of the same image and different captions. Your task is to evaluate the relevance between each
caption provided and the image, which is at the top of the web page.

### Process

- **Checking the image** - In the annotation tool, you will see the image. Please check it carefully. You can enlarge
  it by clicking on the image.
- **Read a caption** - For each image, there will be multiple captions, but the number of captions differs for each
  image. Please, read carefully the caption and based on the image, annotate the relevance between caption and image as
  mentioned above. By selecting one of the options (Yes, No, ?) you will annotate the specific pair of caption and
  image. After providing annotations for each caption (take each caption as an individual task, multiple captions can
  be relevant for one image), click the SUBMIT button at the bottom of the page.

Four examples of retrieval were also shown as part of the instructions, together with the correct annotations and the
rationale.

## How the labels are stored

In the list files of `svo_probes/`, `human evaluation` holds the expert's label per retrieved item (`correct`,
`subject incorrect`, `verb incorrect`, `object incorrect`), and, for the three-annotator lists, `human evaluation 2`
and `human evaluation 3` hold the votes of the two volunteers (1 = yes, -1 = no, 0 = can not tell). A pair counts as
correct when at least two of the three annotators say yes.
