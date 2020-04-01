# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import pathlib
import pkg_resources


with pathlib.Path('./requirements.txt').open() as req:
    install_requires = [str(r) for r in pkg_resources.parse_requirements(req)]


setup(
    name = 'fakeras',
    version = '0.0.1',
    keywords = ('deep learning framework', 'a faker of keras'),
    description = 'fakeras: a faker of keras',
    long_description = 'Fakeras is just a toy for learning deep learning,  '
                       'it maintains the same interface as keras.',
    license = 'Apache 2.0',

    author = 'Eassi Chan',
    author_email = 'eassichan@gmail.com',

    packages = find_packages(
        where='.',
        exclude=(),
        include=('*',)
    ),
    platforms = 'any',
    install_requires = install_requires,
    include_package_data = True,
    package_data={
        'fakeras': [
            'datasets/data/*.npz',
            'datasets/data/*.json'
        ],
    }
)