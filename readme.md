`get_features.py` includes functions from [Wikimania 19](https://github.com/mirrys/Wikimania_19_Tutorial/). `image_similarity_tools.py` was slightly modified from the downloadable and the main notebook modified accordingly, `image_similarity.ipynb` should be run-all-able.

Work still to be done:
1. Display precision and recall values on a pandas dataframe for each algorithm, for each value of n, check variation with increasing n, average and standard deviation given random samples of inputs.
2. Try including the removed 599px wide outliers and verify there is no big change in predictions.
3. Implement PCA to speed up predictions
4. Play around with Mobilenet model's parameters. Compare with other models like inception (after removing graph() from tfeatures, it eats up RAM).

Among many other things
