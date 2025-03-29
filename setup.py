from setuptools import setup, find_packages

setup(
    name="azerbaijani-address-parser",
    version="0.1.0",
    description="Machine learning system for parsing Azerbaijani addresses",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=2.0.1",
        "pandas>=1.3.3",
        "numpy>=1.21.2",
        "scikit-learn>=1.0",
        "joblib>=1.1.0",
        "tqdm>=4.62.3",
        "spacy>=3.2.0",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
    ],
    extras_require={
        "bert": [
            "torch>=1.10.0",
            "transformers>=4.12.5",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)