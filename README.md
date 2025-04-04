How to make this work with arbitrary classes?

What do we have?
    - Doodle - NL descriptions

What do we want?
    - NL -> Doodle completion

Want some vision-aware text embeddings. Want to associate the embedding with the strokes. 

Contrastive Vision Encoder? Not sure if CVE's are specifically for image embeddings, I think we want the opposite. If a CVE is a NL-aware image encoder, then we want a image-aware text encoder.

Take the output of this encoder, append previous strokes -> predict the next stroke

Data description:
    - A sketch is a 3-length tuple, (x, y, lift_pen)
    - If lift_pen == 1, then you lift the pen and just go to the next point directly without drawing a line


Model input description:
    - (delta_x, delta_y, pen_on_paper, pen_off_paper, finished)
    - First two are change in coordinates compared to the previous point

Thoughts:
    - Last three are one-hot vector, so they're just logits for probabilities
    - We can represent the first two as distributions, one value for mu, one for logvar (so all values, positive + negative are allowed. To get the sample, we just exponentiate)


Input to model: (delta_x_mu, delta_x_logvar, delta_y_mu, delta_y_logvar, pen_on_paper, pen_off_paper, finished)

Encoder will take the whole 7-length vector and compress it into a single embedding.

Append the CLIP embedding to the beginning (or anywhere, order doesn't matter as long as it's consistently in the same place)

Pass it through the transformer model as usual

Decoder will take the output and project it out back into the 7 values

Loss:
    - For x, y distributions: Want to measure the error in the two distributions. Honestly for a first iteration, you could just predict the mean itself, no std
    - For pen_on_paper, pen_off_paper, finished -> cross entropy loss (since this is just classification)


TODO:
    - Build Dataset
        - Collect and preprocess all doodles
        - Load the embedding model + calculate embeddings for all the classes
        - EXPERIMENT: Normalize embeddings
        - X: class name embedding + a sequence of strokes of length k, k >= 1
        - Y: strokes all shifted left one + new stroke
    - Build Model
        - Pass strokes through an MLP
        - Concat class name embedding with strokes
        - EXPERIMENT: Add position embeddings to strokes?
        - Transformer Architecture, multiple layers of Self-Attention + MLP
        - Decoder takes the output and puts it back into the length-5 vector
        - Calculate Loss, Optimizer Step, etc.


How do we want to structure the input?

Inputs are between lengths of 49 - 148

Padding/Clipping?
 - Target a certain size of input -> 30?
 - Pad/Clip
 