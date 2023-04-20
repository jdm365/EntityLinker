import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()
    setuptools.setup(
            name='EntityLinker',
            version='0.0.1',
            author='your name',
            author_email='jdm365@georgetown.edu',
            description='Curation of methods for entity linking.',
            long_description=long_description, 
            long_description_content_type='text/markdown', 
            packages=setuptools.find_packages(), 
            python_requires='>=3.6'
            )
