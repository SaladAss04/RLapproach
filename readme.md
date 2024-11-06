### Environment Setup
#### If you're running for the first time:
after cloning the github repo,
`
cd RLapproach
python3 -m venv rla
source rla/bin/activate
`
#### Otherwise:
`
cd RLapproach
source rla/bin/activate
`
#### Perform Training
`
python main.py
`
Training progress will be shown by progress bar, and ablation figures will be generated in the root folder; theres 5 of them.
#### Please notice:
1. I am currently using the DummyEnv, because there's no need for rendering now. If there's discrepancies between the rendered env and dummy, please stick to dummy as it is proven to work.
2. Please change the NUM_ITERATIONS constant in main.py to change the length of the training. Other training-related constants are in components.py, you'll probably have to intensively tweak those for a better result.
