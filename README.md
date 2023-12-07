# distilled Self-Critique
Code for the paper "Distilled Self-Critique of LLMs with Synthetic Data: a Bayesian Perspective", https://arxiv.org/abs/2312.01957 

## Synthetic Data Generation (self-critique)

It requires the ollama server for faster inference. You can get it from [here](https://ollama.ai).
Once you have the ollama server running, you can run the following command to generate synthetic data:

```bash
python generate_synthetic.py
```

where `generate_synthetic.py` is located in each of the three experiment folders from the paper:

* `safety`: avoiding generating harmful responses.
* `sentiments`: generating positive sentiment movie reviews, even when prompted to be negative.
* `privacy`: generating news without revealing the identity of the person (measured with a proxy NER model).

Inside each `generate_synthetic.py` file, you will find the exact prompts for each experiment, in particular:

* The original prompt for each example.
* The critique prompt.
* The review prompt.

These are the steps for the Gibbs sampling chain described in the paper. In these experiments, we found that with just one step of each of the previous steps was sufficient. This avoids taking too long to generate the synthetic data.

## Model Distillation (SFT)

Once you have generated the synthetic data, you can finetune the model over a training split of the synthetic data. You may filter the data first using the reward scores of the revised and original sample (acceptance step in the paper). This is recommended as it improves the results over the test set of prompts.

Run the notebook [`run_distillation.ipynb`](run_distillation.ipynb) to self-distil the model on the synthetic data generated in the previous seep. The notebook uses the `sentiment` task an example, but it can be easily adapted to the other tasks.

### Citation

If you find this work useful, please consider citing with

```
@misc{gallego2023distilled,
      title={Distilled Self-Critique of LLMs with Synthetic Data: a Bayesian Perspective}, 
      author={Victor Gallego},
      year={2023},
      eprint={2312.01957},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

