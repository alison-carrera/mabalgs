clean:
	rm -rf pytest_cache/
	rm -rf dist/
	rm -rf mabalgs.egg-info/

build:
	python setup.py sdist

deploy: clean build
	twine upload dist/*