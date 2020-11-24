'''
setup.py for cotk
'''
import sys
import os
from setuptools import setup, find_packages

setup(
	name='eva',
	version='0.0.1',
	packages=find_packages(exclude=[]),
	license='Apache',
	description='Evaluation Toolkits for evaluating open-ended story generation',
	long_description=open('README.md', encoding='UTF-8').read(),
	long_description_content_type="text/markdown",
	install_requires=[
		'numpy>=1.19.2',
		'nltk>=3.4',
		'tqdm>=4.30',
		'checksumdir>=1.1',
		'bert_score>=0.3.6',
		'moverscore>=0.95',
		'rouge>=1.0.0',
		'tensorflow-gpu>=1.15.0,<2.0.0',
		'torch>=1.4.0',
		'transformers>=3.1.0',
		'bert_score>=0.3.6',
		'requests>=2.23.0',
		'kendall-w>=1.0.0',
		'spacy>=2.3.2',
	],
	},
	entry_points={
	},
	include_package_data=True,
	url='https://github.com/thu-coai/OpenEVA',
	author='thu-coai',
	author_email='thu-coai-developer@googlegroups.com',
	python_requires='>=3.7',
)
