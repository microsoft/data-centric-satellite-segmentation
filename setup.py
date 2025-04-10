#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Akram Zaytar",
    author_email='akramzaytar@microsoft.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Utilities for data preparation, training, and evaluation for multiple remote sensing datasets.",
    entry_points={
        'console_scripts': [
            'mveo_benchmarks=mveo_benchmarks.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='mveo_benchmarks',
    name='mveo_benchmarks',
    packages=find_packages(include=['mveo_benchmarks', 'mveo_benchmarks.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/akramz/mveo_benchmarks',
    version='0.1.0',
    zip_safe=False,
)
