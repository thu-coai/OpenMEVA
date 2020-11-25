'''
setup.py for cotk
'''
import sys
import os
from setuptools import setup, find_packages
import setuptools

setup(
	name='eva',
	version='0.0.1',
	packages=find_packages(exclude=[]),
	description='Evaluation Toolkits for evaluating open-ended story generation',
	long_description=open('README.md', encoding='UTF-8').read(),
	long_description_content_type="text/markdown",
	entry_points={
	},
	include_package_data=True,
	url='https://github.com/thu-coai/OpenEVA',
	packages=setuptools.find_packages(),
	author='thu-coai',
	author_email='thu-coai-developer@googlegroups.com',
	python_requires='>=3.7',
)
