from setuptools import setup

setup(
    name="BrokerPython",
    version="0.1",
    py_modules=['main'],
    install_requires=[
        'Click',
        'grpcio',
        'grpcio-tools',
        'gym',
        'h5py==2.10.0',
        # 'keras',
        # 'keras-rl',
        'mypy',
        #'mypy-protbuf',
        'numpy==1.18.5',
        'pandas',
        'PyDispatcher',
        'protobuf',
        'scikit-learn',
        'beautifulsoup4',
        'lxml',
        'tensorflow',
        'tensorforce'
        ],
    extras_require = {
        'visualize':[
            'jupyter',
            'tensorboard'
            ],
        'tests':[
            'pytest',
            'pytest-watch'
            ]
        },
    entry_points='''
        [console_scripts]
        agent=main:cli
    ''')
