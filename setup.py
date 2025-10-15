
from setuptools import setup, find_packages

setup(
    name="TeluguSentiment",
    version="0.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Telugu Sentiment Analysis Python package using MuRIL",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "transformers",
        "emoji",
        "pandas",
        "openpyxl"
    ],
)
