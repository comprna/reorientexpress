import setuptools

setuptools.setup(
    name = 'reorientexpress',
    version = '0.105',
    scripts = ['reorientexpress.py'],
    install_requires=[
        'numpy>=1.15.3',
        'Keras>=2.2.4',
        'pandas>=0.23.4',
        'scikit_learn>=0.20.1',
        'tensorflow>=1.11.0'
        ],
    author = 'Angel Ruiz Reche',
    author_email = 'angelrure@gmail.com',
    description = 'Script used to build, test and use models that predict the orientation of cDNA reads',
    url = 'https://github.com/angelrure/reorientexpress',
    packages=setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3"
    ]
)