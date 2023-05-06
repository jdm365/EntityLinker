all:
	rm -rf build dist *.egg-info
	python setup.py install

clean:
	rm -rf build dist *.egg-info
