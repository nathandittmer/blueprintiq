run:
	python -m blueprintiq.cli run --config blueprintiq/config/default.yaml

gen-data:
	python -m blueprintiq.cli gen-data --config blueprintiq/config/default.yaml

viz-data:
	python -m blueprintiq.datasets.viz_coco_sample

test-detector:
	python -m blueprintiq.models.test_detector

train-detector:
	python -m blueprintiq.training.train_detector

eval-detector:
	python -m blueprintiq.training.eval_detector

viz-predictions:
	python -m blueprintiq.training.viz_predictions