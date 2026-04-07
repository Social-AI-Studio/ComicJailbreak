# ComicJailbreak

ComicJailbreak introduces a comic-based jailbreak dataset to evaluate whether MLLMs uphold safety policy when harmful goals are embedded in visual narrative.

## Dataset 
### Dataset Creation
To create ComicJailbreak, run the following code:
```bash
pip install -r requirements.txt

python create_dataset.py --type article
```

- You can extend the dataset by creating a `csv` file that looks like this:

| Article | Speech | Instruction | Message | Code |
| --- | --- | --- | --- | --- |
| Your prompt | ... | ... | ... | ... | 

## Remarks
We have released the ComicJailbreak dataset, and will provide the inference and evaluation code when they are ready. Stay tuned...

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