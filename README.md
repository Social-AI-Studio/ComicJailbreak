# ComicJailbreak

ComicJailbreak introduces a comic-based jailbreak dataset to evaluate whether MLLMs uphold safety policy when harmful goals are embedded in visual narrative.

## Dataset 
### Dataset Creation
To create ComicJailbreak, run the following code (we are using `uv` for the environment setup):
```bash
uv sync

uv run python create_dataset.py --type article
```

- You can extend the dataset by creating a `csv` file that looks like this:

| Article | Speech | Instruction | Message | Code |
| --- | --- | --- | --- | --- |
| Your prompt | ... | ... | ... | ... | 

For paraphrasing, run the following code:
```bash
uv run python paraphrasing --goals <your_path.csv> --type all
```

## Experiments
If you were to use the API inferences, please include `.env` file to store your API keys:
```text
OPENROUTER_API=<YOUR_API_KEY>
```

We have included the prompts to perform the attack and defenses, packaged into shell scripts:
```bash
# For ComicJailbreak attack experiments
bash attack.sh              # Local inference
bash openrouter_attack.sh   # API inference (we are using OpenRouter as the provider)

# For defenses against ComicJailbreak
bash defense.sh             # Local inference
bash openrouter_defense.sh  # API inference (we are using OpenRouter as the provider)

# Script for evaluation
bash eval.sh                # Local inference
```

## Citation
If you find this work useful in your research, please cite the following paper:
```bibtex
@article{tan2026structured,
  title={Structured Visual Narratives Undermine Safety Alignment in Multimodal Large Language Models},
  author={Tan, Rui Yang and Hu, Yujia and Lee, Roy Ka-Wei},
  journal={arXiv preprint arXiv:2603.21697},
  year={2026}
}
```