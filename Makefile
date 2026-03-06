run:
	python -m blueprintiq.cli run --config blueprintiq/config/default.yaml

gen-data:
	python -m blueprintiq.cli gen-data --config blueprintiq/config/default.yaml

viz-data:
	python -m blueprintiq.datasets.viz_coco_sample