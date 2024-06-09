from setuptools import setup, find_packages

setup(
    name='ytsum',
    version='0.1.5',
    packages=find_packages(),
    install_requires=[
        'torch==2.1.1+cu118',
        'requests==2.32.3',
        'pytube==15.0.0',
        'transformers==4.36.2',
        'langchain==0.2.3',
        'langchain_community==0.2.4',
        'langchain_together==0.1.3'        
    ],
    entry_points={
        'console_scripts': [],
    },
    author='Arihant Tripathi',
    author_email='tarihant2001@gmail.com',
    description='Summarize YouTube video instantly with the power of distill-whisper and Mixtral-8B',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Siris2314/ytsum',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
